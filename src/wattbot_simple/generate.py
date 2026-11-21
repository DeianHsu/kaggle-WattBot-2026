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
SYSTEM_PROMPT = (
    "只根据给定证据，用中文简洁回答问题。"
    "在事实后用 [Evidence_0] 等标签引用，只能使用提供的标签。"
    "证据不足时明确说明，并返回空的 citations 列表。"
    '输出 JSON：{"answer": "回答 [Evidence_0]", '
    '"citations": ["Evidence_0"]}。'
)


def retrieve_evidence(question, retriever, llm, answer_unit=""):
    try:
        plan = llm.invoke([
            SystemMessage(content=SEARCH_PROMPT),
            HumanMessage(content=f"{question}\n指定单位：{answer_unit}"),
        ])
    except OutputParserException:
        logger.warning("检索规划无法解析，使用原问题检索")
        plan = {}

    subqueries = plan.get("queries", []) if isinstance(plan, dict) else []
    if not isinstance(subqueries, list) or any(
        not isinstance(query, str) for query in subqueries
    ):
        logger.warning("忽略格式异常的检索字段 queries")
        subqueries = []
    subqueries = [query.strip() for query in subqueries if query.strip()]
    queries = list(dict.fromkeys([question, *subqueries]))
    return retrieve_queries(retriever, queries)


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


# 根据问题和图文证据生成回答，并将模型引用还原成真实来源。
def generate_answer(question, retriever, llm):
    llm = llm.with_structured_output(method="json_mode")
    evidence = retrieve_evidence(question, retriever, llm)
    evidence_blocks, citation_map = build_evidence_blocks(evidence)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content_blocks=[
                create_text_block(f"question:{question}"),
                *evidence_blocks,
            ]
        )
    ]
    llm_answer = llm.invoke(messages)
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
