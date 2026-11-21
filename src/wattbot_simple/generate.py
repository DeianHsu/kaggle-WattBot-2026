# 规划论文检索查询，将文本、表格和原图组装成带标签的证据。
# 根据证据生成并校验答案，记录标签对应的论文来源、页码和文本块标识。

import json
from pathlib import Path

from docling_core.types import DoclingDocument
from docling_core.types.doc import RefItem
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
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


# 规定正式作答的答案字段、数值格式和证据不足时的填写方式。
ANSWER_PROMPT = """Answer using only the supplied evidence and return JSON.
Write answer and explanation in English. Put only the direct answer in
answer_value. Use explanation to briefly show how the evidence supports it.
Convert numeric answers to answer_unit even if the question mentions a
different unit. For 2 million kWh, write "2000000" if the unit is kWh or "2"
if it is million kWh. Omit units, thousands separators, and words such as
million from numeric values.
For terms, use the shortest complete wording from the source, keeping
necessary modifiers and conjunctions. Do not automatically wrap multiple
terms in parentheses.
For true or false questions, use "1" for true and "0" for false.
Write multiple values in the order requested, and ranges from lower to
upper bound. Use a string such as "(0.5,1.65)" instead of a JSON array.
In citations, list only supplied evidence labels that support the answer.
If the question can be answered, include at least one citation. In
supporting_materials, quote the source verbatim in its original language
or give the location of the supporting figure or table.
If the evidence is insufficient, set answer_value to "is_blank", leave
citations empty, and briefly explain what evidence is missing in answer
and explanation.
All output fields are strings except citations, which is a list of strings.
An answer_unit of "is_blank" means the question has no specified unit.

Example JSON
{
  "answer": "The sample contained 12 items.",
  "answer_value": "12",
  "citations": ["Evidence_0"],
  "supporting_materials": "The sample contained twelve items.",
  "explanation": "The cited sentence gives the sample count."
}
"""


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


# 为一道题检索证据并生成答案，返回校验后的答案及证据标签的映射。
def generate_answer(question, retriever, llm, answer_unit):
    evidence = retrieve_evidence(question, retriever, llm, answer_unit=answer_unit)
    evidence_blocks, citation_map = build_evidence_blocks(evidence)
    question_block = create_text_block(
        f"Question: {question}\n"
        f"Requested unit: {answer_unit}"
    )
    messages = [
        SystemMessage(content=ANSWER_PROMPT),
        HumanMessage(content_blocks=[question_block, *evidence_blocks]),
    ]
    # 解析或校验输出失败时，把错误反馈给模型，最多纠正一次。
    for attempt in range(2):
        try:
            result = llm.invoke(messages)
            result = normalize_prediction_result(result, citation_map)
            break
        except OutputParserException as exc:
            if attempt == 1:
                raise
            messages.extend([
                AIMessage(
                    content=exc.llm_output
                    or "The previous response was not valid JSON."
                ),
                HumanMessage(content=(
                    f"Please correct the previous output: {exc}\n"
                    f"Available evidence labels: {', '.join(citation_map)}.\n"
                    "Use the original question and evidence to return complete JSON. "
                    "Cite only evidence that supports the answer. "
                    "If the evidence is insufficient, set answer_value to is_blank."
                )),
            ])

    return result, citation_map


# 校验模型输出并去除字符串首尾空白，保留错误输出供模型纠正。
def normalize_prediction_result(result, citation_map):
    result_json = json.dumps(result, ensure_ascii=False)
    if not isinstance(result, dict):
        raise OutputParserException(
            "The output must be a JSON object.", llm_output=result_json
        )

    for field in ("answer", "answer_value", "supporting_materials", "explanation"):
        if field not in result or not isinstance(result[field], str):
            raise OutputParserException(
                f"{field} is missing or is not a string.", llm_output=result_json
            )
        result[field] = result[field].strip()

    if "citations" not in result or not isinstance(result["citations"], list):
        raise OutputParserException(
            "citations must be a list of strings.", llm_output=result_json
        )
    if any(
        not isinstance(label, str) or label not in citation_map
        for label in result["citations"]
    ):
        raise OutputParserException(
            "The citations list contains an invalid evidence label.",
            llm_output=result_json,
        )
    if not all(result[field] for field in ("answer", "answer_value", "explanation")):
        raise OutputParserException(
            "answer, answer_value, and explanation must be non-empty.",
            llm_output=result_json,
        )
    if result["answer_value"] != "is_blank" and (
        not result["citations"] or not result["supporting_materials"]
    ):
        raise OutputParserException(
            "An answerable question requires citations and supporting_materials.",
            llm_output=result_json,
        )
    return result
