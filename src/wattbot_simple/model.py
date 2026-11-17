import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

ROOT_DIR = Path(__file__).resolve().parents[2]


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
