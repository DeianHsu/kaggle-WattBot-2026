# 将检索到的文本、表格和原图组装成带证据标签的模型输入。
# 根据图文证据生成回答，并把引用标签还原为论文来源、页码和文本块标识。

import json
import logging
from pathlib import Path
from docling_core.types import DoclingDocument
from docling_core.types.doc import RefItem
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages.content import create_text_block
from wattbot_simple.model import load_image_block
from wattbot_simple.retrieve import retrieve_queries

logger = logging.getLogger(__name__)

SEARCH_PROMPT = """为论文检索规划简短查询。原问题会单独检索。
比较、计算和跨论文问题按回答所需的独立事实拆分，每项事实写一条查询，明确对象和指标。
保留各事实自己的年份、单位、范围和条件。沿用问题中的术语，必要时补一个常用英文术语写法。
只给检索短语，禁止猜测答案、未给出的数值或论文。单一事实已表述清楚时返回空列表。
输出 JSON：{"queries": ["short search phrase"]}。"""
FOLLOW_UP_PROMPT = """首轮证据若缺少回答所必需的事实，在 JSON 的 missing_queries 中为每个缺项列一条简短检索查询。
保留各缺项的对象、指标和条件，可用已有证据中的论文名或术语帮助定位。证据齐全时返回空列表。
补查后的回答依据更新后的证据，缺少关键事实时仍按原规则说明证据不足。"""
SYSTEM_PROMPT = (
    "只根据给定证据，用中文简洁回答问题。"
    "在事实后用 [Evidence_0] 等标签引用，只能使用提供的标签。"
    "证据不足时明确说明，并返回空的 citations 列表。"
    '输出 JSON：{"answer": "回答 [Evidence_0]", '
    '"citations": ["Evidence_0"]}。'
)


def _read_queries(result, field):
    queries = result.get(field, []) if isinstance(result, dict) else []
    if not isinstance(queries, list) or any(not isinstance(query, str) for query in queries):
        logger.warning("忽略格式异常的检索字段 %s", field)
        return []
    return list(dict.fromkeys(query.strip() for query in queries if query.strip()))


def generate_with_retrieval(question, retriever, llm, draft_answer, answer_unit=""):
    try:
        plan = llm.invoke([
            SystemMessage(content=SEARCH_PROMPT),
            HumanMessage(content=f"{question}\n指定单位：{answer_unit}"),
        ])
    except OutputParserException:
        logger.warning("检索规划无法解析，使用原问题检索")
        plan = {}

    queries = list(dict.fromkeys([question, *_read_queries(plan, "queries")]))
    evidence = retrieve_queries(retriever, queries)
    result, citation_map = draft_answer(evidence)
    asked = {query.casefold() for query in queries}
    follow_up = [
        query for query in _read_queries(result, "missing_queries")
        if query.casefold() not in asked
    ]
    if follow_up:
        updated = retrieve_queries(retriever, follow_up, previous=evidence)
        previous_ids = {chunk.metadata["chunk_id"] for chunk in evidence}
        if any(chunk.metadata["chunk_id"] not in previous_ids for chunk in updated):
            result, citation_map = draft_answer(updated)
    return result, citation_map


# 将检索结果组装成图文消息，并建立证据标签到原始来源的映射。
def build_evidence_blocks(evidence):
    evidence_blocks = []
    citation_map = {}

    for num, chunk in enumerate(evidence):
        # 将检索文字和论文标识放在同一个证据标签下。
        label = f"Evidence_{num}"
        metadata = chunk.metadata
        items = json.loads(metadata["dl_meta"])["doc_items"]
        visual_items = [item for item in items if item["label"] in {"table", "picture"}]
        text = chunk.page_content
        if visual_items:
            text = "请结合随附原图核对图表信息。\n" + text

        evidence_blocks.append(
            create_text_block(
                f"[{label}]\n"
                f"论文标识：{Path(metadata['source']).stem}\n"
                f"{text}"
            )
        )

        # 从原始元素收集页码，记录标签对应的真实来源。
        pages = sorted({
            prov["page_no"]
            for item in items
            for prov in item["prov"]
        })
        citation_map[label] = {
            "chunk_id": metadata["chunk_id"],
            "source": metadata["source"],
            "pages": pages,
        }

        if not visual_items:
            continue

        # 按原始元素标识取回完整图表，补充检索文字没有包含的内容。
        document_path = Path(metadata["document_path"])
        pdf_document = DoclingDocument.load_from_json(document_path)
        for item in visual_items:
            element = RefItem(cref=item["self_ref"]).resolve(pdf_document)
            if item["label"] == "table":
                # 解析文字作为辅助，原表图片用于核对数据和行列对应关系。
                evidence_blocks.append(
                    create_text_block(
                        "表格解析文字可能缺失或错位，请结合随后原图核对数据和行列。\n"
                        + element.export_to_html(doc=pdf_document)
                    )
                )
            # 图表共用图片读取与编码逻辑，让模型直接查看原图。
            evidence_blocks.append(load_image_block(element.image, document_path.parent))
    return evidence_blocks, citation_map


def _draft_answer(question, evidence, llm):
    evidence_blocks, citation_map = build_evidence_blocks(evidence)

    messages = [
        SystemMessage(
            content=f"{SYSTEM_PROMPT}\n{FOLLOW_UP_PROMPT}",
        ),
        HumanMessage(
            content_blocks=[
                create_text_block(f"question:{question}"),
                *evidence_blocks,
            ]
        )
    ]
    return llm.invoke(messages), citation_map


# 根据问题和图文证据生成回答，并将模型引用还原成真实来源。
def generate_answer(question, retriever, llm):
    llm = llm.with_structured_output(method="json_mode")
    llm_answer, citation_map = generate_with_retrieval(
        question, retriever, llm,
        lambda evidence: _draft_answer(question, evidence, llm),
    )
    # 去除重复引用标签，并恢复每条引用的论文来源和页码。
    references = [
        {
            "label": label,
            **citation_map[label],
        }
        for label in dict.fromkeys(llm_answer["citations"])
    ]
    result = {
        "answer": llm_answer["answer"],
        "references": references,
    }

    return result
