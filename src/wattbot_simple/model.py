import os
from base64 import b64encode
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.messages.content import create_image_block, create_text_block
from langchain_openai import ChatOpenAI

ROOT_DIR = Path(__file__).resolve().parents[2]
PICTURE_DESCRIPTION_PROMPT = (
    "Write a short English description of this research figure in at most 60 words. "
    "Include its visible subject, figure type, axis labels, units, legend entries, "
    "and named components when present. Omit unreadable labels, guessed values, "
    "and conclusions that are not shown in the image."
)


# 读取访问密钥并创建 DeepSeek 问答模型，默认关闭思考。
def create_llm(thinking="none"):
    # 从项目的环境配置文件加载模型访问密钥。
    load_dotenv(ROOT_DIR / ".env")
    # 按强度开关思考；开启时采用接口默认输出上限，避免原来的短上限截断思考。
    extra_body = {
        "thinking": {"type": "disabled" if thinking == "none" else "enabled"},
    }
    if thinking == "none":
        extra_body["max_tokens"] = 2048

    # 复用同一个模型配置，只将思考强度交给调用方设置。
    return ChatOpenAI(
        model="deepseek-flash",
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
        temperature=0,
        reasoning_effort=thinking,
        extra_body=extra_body,
        timeout=120,
    )


def create_picture_vlm():
    load_dotenv(ROOT_DIR / ".env")
    model_name = os.getenv("VLM_MODEL", "").strip()
    base_url = os.getenv("VLM_BASE_URL", "").strip()
    if not model_name or not base_url:
        raise ValueError("启用图片描述前，请配置 VLM_MODEL 和 VLM_BASE_URL")

    return ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key=os.getenv("VLM_API_KEY") or "local",
        use_responses_api=False,
        temperature=0,
        max_completion_tokens=192,
        timeout=120,
        max_retries=0,
    )


def load_image_block(image_path, mime_type):
    return create_image_block(
        base64=b64encode(image_path.read_bytes()).decode("ascii"),
        mime_type=mime_type,
    )


def describe_picture(vlm, image_path, mime_type):
    message = HumanMessage(content_blocks=[
        create_text_block(PICTURE_DESCRIPTION_PROMPT),
        load_image_block(image_path, mime_type),
    ])
    description = vlm.invoke([message]).text.strip()
    if not description:
        raise ValueError("视觉语言模型返回了空描述")
    return description
