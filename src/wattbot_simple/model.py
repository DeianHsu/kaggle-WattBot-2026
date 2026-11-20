# 创建问答和图表描述共用的模型客户端，组装图文输入。

import os
from base64 import b64encode
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.messages.content import create_image_block, create_text_block
from langchain_openai import ChatOpenAI

ROOT_DIR = Path(__file__).resolve().parents[2]
VISUAL_DESCRIPTION_PROMPT = (
    "Describe this figure or table in one short English paragraph for search. "
    "Include the subject, visible labels and units, and any legend entries, "
    "row names, or column names. Use only details visible in the image."
)


def create_llm(thinking="none"):
    load_dotenv(ROOT_DIR / ".env")
    # 开启思考时沿用接口默认输出上限，为推理留出空间。
    extra_body = {
        "thinking": {"type": "disabled" if thinking == "none" else "enabled"},
    }
    if thinking == "none":
        extra_body["max_tokens"] = 2048

    return ChatOpenAI(
        model="deepseek-flash",
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
        temperature=0,
        reasoning_effort=thinking,
        extra_body=extra_body,
        timeout=120,
    )


def load_image_block(image, document_dir):
    uri = str(image.uri)
    if uri.startswith("data:"):
        return create_image_block(url=uri)
    image_path = document_dir / uri
    return create_image_block(
        base64=b64encode(image_path.read_bytes()).decode("ascii"),
        mime_type=image.mimetype,
    )


def describe_image(vlm, image_block):
    message = HumanMessage(content_blocks=[
        create_text_block(VISUAL_DESCRIPTION_PROMPT),
        image_block,
    ])
    return vlm.invoke([message]).text.strip()
