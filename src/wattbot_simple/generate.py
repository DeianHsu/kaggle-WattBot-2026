# 将检索到的文本、表格和原图组装成带证据标签的模型输入。
# 规划论文检索查询，并记录各证据标签对应的论文来源、页码和文本块标识。

import json
from pathlib import Path

from docling_core.types import DoclingDocument
from docling_core.types.doc import RefItem
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages.content import create_text_block

from wattbot_simple.model import load_image_block
from wattbot_simple.retrieve import retrieve_queries


SEARCH_PROMPT = """Plan short English queries to search research papers.
The original question will be searched separately.
For comparison, calculation, and cross-paper questions, write one query for
each independent fact needed to answer. Name the subject and metric in each.
Keep each fact's year, units, scope, and conditions. Retain the question's
terms and add a common English variant when useful.
Return only search phrases. Do not guess answers, values, or papers that
were not given. Return an empty list if the question clearly asks for a
single fact.
Return JSON in this format.
{"queries": ["short search phrase"]}"""


def retrieve_evidence(question, retriever, llm, answer_unit=""):
    messages = [
        SystemMessage(content=SEARCH_PROMPT),
        HumanMessage(content=f"{question}\nRequested unit: {answer_unit}"),
    ]
    queries = [question]
    try:
        plan = llm.invoke(messages)
    except OutputParserException:
        return retrieve_queries(retriever, queries)

    subqueries = plan.get("queries") if isinstance(plan, dict) else None
    if isinstance(subqueries, list) and all(
        isinstance(query, str) for query in subqueries
    ):
        queries.extend(query.strip() for query in subqueries if query.strip())
    queries = list(dict.fromkeys(queries))
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
            text = "Check the original image to verify the details.\n" + text

        evidence_blocks.append(
            create_text_block(
                f"[{label}]\n"
                f"Paper ID: {Path(metadata['source']).stem}\n"
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
                        "The extracted table text may be incomplete or misaligned. "
                        "Use the original image below to verify the values, rows, "
                        "and columns.\n"
                        + element.export_to_html(doc=pdf_document)
                    )
                )
            # 图表共用图片读取与编码逻辑，让模型直接查看原图。
            evidence_blocks.append(load_image_block(element.image, document_path.parent))
    return evidence_blocks, citation_map
