# 使用已有索引运行单题问答示例，展示答案、引用论文和页码。

from pathlib import Path
from wattbot_simple.cost import record_cost
from wattbot_simple.generate import generate_answer
from wattbot_simple.index import open_chroma_store
from wattbot_simple.model import create_llm
from wattbot_simple.retrieve import create_evidence_retriever


ROOT_DIR = Path(__file__).resolve().parents[2]


def answer_question(question, retriever):
    # 单题调试以问题正文为标识，共用批量预测的成本记录流程。
    with record_cost(ROOT_DIR / "answer_cost.json", question):
        llm = create_llm()
        answer_result = generate_answer(question, retriever, llm)
    return answer_result


def print_answer(answer_result):
    print(answer_result["answer"])
    for reference in answer_result["references"]:
        filename = Path(reference["source"]).name
        pages = ", ".join(map(str, reference["pages"]))
        print(f"[{reference['label']}] {filename}, PDF pages: {pages}")


def answer():
    chroma_store = open_chroma_store()
    retriever = create_evidence_retriever(chroma_store)
    question = "How does the framework coordinate cooling and computing in AI datacenters?"
    answer_result = answer_question(question, retriever)
    print_answer(answer_result)


if __name__ == "__main__":
    answer()
