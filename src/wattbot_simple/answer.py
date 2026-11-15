from pathlib import Path
from wattbot_simple.cost import record_cost
from wattbot_simple.generate import generate_answer
from wattbot_simple.index import open_chroma_store
from wattbot_simple.model import get_llm
from wattbot_simple.retrieve import create_evidence_retriever


ROOT_DIR = Path(__file__).resolve().parents[2]


# 检索单个问题的证据，并调用模型生成带引用的回答。
def answer_question(question, retriever):
    # 单题调试以问题正文为标识，共用批量预测的成本记录流程。
    with record_cost(ROOT_DIR / "answer_cost.json", question):
        # 先取回相关证据，再交给回答流程组织图文请求。
        evidence = retriever.invoke(question)
        llm = get_llm()
        answer_result = generate_answer(question, evidence, llm)
    return answer_result


# 输出答案正文，以及每条引用对应的论文和页码。
def print_answer(answer_result):
    print(answer_result["answer"])
    # 将引用中的文件路径和页码整理成便于阅读的格式。
    for reference in answer_result["references"]:
        filename = Path(reference["source"]).name
        pages = ", ".join(map(str, reference["pages"]))
        print(f"[{reference['label']}] {filename}，PDF 页码：{pages}")


# 打开已有索引，运行并展示一条示例问答。
def answer():
    # 为本次问答准备索引和检索器。
    chroma_store = open_chroma_store()
    retriever = create_evidence_retriever(chroma_store)
    # 对示例问题生成答案，并显示对应的证据来源。
    question = "How does the framework coordinate cooling and computing in AI datacenters?"
    answer_result = answer_question(question, retriever)
    print_answer(answer_result)


if __name__ == "__main__":
    answer()
