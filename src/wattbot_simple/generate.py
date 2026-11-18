import json
from pathlib import Path
from docling_core.types import DoclingDocument
from docling_core.types.doc import RefItem
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages.content import create_text_block
from wattbot_simple.model import load_image_block

SYSTEM_PROMPT = (
    "只根据给定证据，用中文简洁回答问题。"
    "在事实后用 [Evidence_0] 等标签引用，只能使用提供的标签。"
    "证据不足时明确说明，并返回空的 citations 列表。"
    '输出 JSON：{"answer": "回答 [Evidence_0]", '
    '"citations": ["Evidence_0"]}。'
)


# 将检索结果组装成图文消息，并建立证据标签到原始来源的映射。
def build_evidence_blocks(evidence):
    evidence_blocks = []
    citation_map = {}

    for num, chunk in enumerate(evidence):
        # 为当前证据分配引用标签，并把正文和论文标识一起交给模型。
        label = f"Evidence_{num}"
        metadata = chunk.metadata
        text = chunk.page_content
        if "picture_description_model" in metadata:
            text = "模型生成的图片描述，用于定位。涉及图中事实时请核对随后原图。\n" + text

        evidence_blocks.append(
            create_text_block(
                f"[{label}]\n"
                f"论文标识：{Path(metadata['source']).stem}\n"
                f"{text}"
            )
        )

        # 从原始元素收集页码，记录标签对应的真实来源。
        items = json.loads(metadata["dl_meta"])["doc_items"]
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

        # 找出块中关联的表格和图片；纯文字块无需加载完整解析文档。
        visual_items = [
            item for item in items
            if item["label"] in {"table", "picture"}
        ]
        if not visual_items:
            continue

        # 按原始元素标识取回完整图表，补充检索文字没有包含的内容。
        pdf_document = DoclingDocument.load_from_json(metadata["document_path"])
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
            image_path = Path(metadata["document_path"]).parent / str(element.image.uri)
            evidence_blocks.append(
                load_image_block(image_path, element.image.mimetype)
            )
    return evidence_blocks, citation_map


# 根据问题和图文证据生成回答，并将模型引用还原成真实来源。
def generate_answer(question, evidence, llm):
    # 构造证据内容和引用映射，再组织本次问答消息。
    evidence_blocks, citation_map = build_evidence_blocks(evidence)

    messages = [
        SystemMessage(
            content=SYSTEM_PROMPT,
        ),
        HumanMessage(
            content_blocks=[
                create_text_block(f"question:{question}"),
                *evidence_blocks,
            ]
        )
    ]
    # 要求模型返回可解析的答案和引用列表。
    llm_answer = llm.with_structured_output(method="json_mode").invoke(messages)
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
