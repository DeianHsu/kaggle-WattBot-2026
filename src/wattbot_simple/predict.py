# 批量生成比赛提交文件，按题保存答案和成本，支持从已有结果续跑。

import argparse
import csv
import json
from pathlib import Path

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages.content import create_text_block

from wattbot_simple.cost import record_cost
from wattbot_simple.generate import build_evidence_blocks, retrieve_evidence
from wattbot_simple.index import open_chroma_store
from wattbot_simple.model import create_llm
from wattbot_simple.retrieve import create_evidence_retriever


ROOT_DIR = Path(__file__).resolve().parents[2]
COLUMNS = (
    "id", "question", "answer", "answer_value", "answer_unit",
    "ref_id", "ref_url", "supporting_materials", "explanation",
)
SYSTEM_PROMPT = """Answer using only the supplied evidence and return JSON.
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


def _draft_prediction(question, evidence, llm):
    evidence_blocks, citation_map = build_evidence_blocks(evidence)
    question_block = create_text_block(
        f"Question: {question['question']}\n"
        f"Requested unit: {question['answer_unit']}"
    )
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
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


# 完成一道比赛题的检索与回答，并整理成提交文件的一行。
def predict_question(question, retriever, llm, paper_urls):
    evidence = retrieve_evidence(
        question["question"], retriever, llm,
        answer_unit=question["answer_unit"],
    )
    result, citation_map = _draft_prediction(question, evidence, llm)
    answer = result["answer"]
    value = result["answer_value"]
    explanation = result["explanation"]
    supporting_materials = result["supporting_materials"]
    if value == "is_blank":
        # 无法回答时，用比赛规定的缺失标记统一填写证据字段。
        ref_id = ref_url = supporting_materials = "is_blank"
    else:
        # 将证据标签合并为论文标识，再从元数据中取得对应链接。
        paper_ids = list(dict.fromkeys(
            Path(citation_map[label]["source"]).stem
            for label in result["citations"]
        ))
        ref_id = json.dumps(paper_ids)
        ref_url = json.dumps([paper_urls[paper_id] for paper_id in paper_ids])

    # 按提交列组织结果，题号、问题和单位直接沿用输入。
    return {
        "id": question["id"],
        "question": question["question"],
        "answer": answer,
        "answer_value": value,
        "answer_unit": question["answer_unit"],
        "ref_id": ref_id,
        "ref_url": ref_url,
        "supporting_materials": supporting_materials,
        "explanation": explanation,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("questions", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ids", nargs="+")
    parser.add_argument(
        "--thinking", choices=("none", "low", "high", "max"), default="none",
        help="思考强度，默认 none 关闭思考，max 开启最高强度",
    )
    return parser.parse_args()


# 读取已有预测文件，确认记录完整后返回按保存顺序排列的题号。
def read_completed_ids(output_path):
    done_ids = []
    if output_path.exists() and output_path.stat().st_size > 0:
        with output_path.open(encoding="utf-8-sig", newline="") as file:
            lines = file.readlines()
        if not lines or not lines[-1].endswith("\n"):
            raise ValueError("已有输出末尾缺少换行，可能未写完整，请检查后重试")

        reader = csv.DictReader(lines, strict=True)
        if reader.fieldnames != list(COLUMNS):
            raise ValueError("已有输出的列名或顺序不符合提交格式")
        for number, row in enumerate(reader, start=1):
            if None in row or any(
                row[column] is None or not row[column].strip()
                for column in COLUMNS
            ):
                raise ValueError(f"已有输出第 {number} 条记录字段不完整，请检查后重试")
            done_ids.append(row["id"])
    return done_ids


# 默认按题库顺序处理。指定 --ids 时按传入的题号顺序处理。
def predict(questions_path, output_path, ids=None, chroma_store=None, thinking="none"):
    with questions_path.open(encoding="utf-8-sig", newline="") as file:
        questions = list(csv.DictReader(file))
    if ids:
        questions_by_id = {question["id"]: question for question in questions}
        questions = [questions_by_id[question_id] for question_id in ids]
    if not questions or len({q["id"] for q in questions}) != len(questions):
        raise ValueError("题库至少要有一道题，题号不能重复")

    done_ids = read_completed_ids(output_path)

    # 续跑时，已有答案必须从第一题起连续对应本次的处理顺序。
    if done_ids != [q["id"] for q in questions[:len(done_ids)]]:
        raise ValueError("已有输出与本次题目顺序不一致")

    start = len(done_ids)
    if start == len(questions):
        print("这批题目已经全部完成")
        return

    # 加载论文标识与链接的对应关系，供生成提交引用时使用。
    with (ROOT_DIR / "input" / "metadata.csv").open(
        encoding="utf-8-sig", newline=""
    ) as file:
        paper_urls = {row["id"]: row["url"] for row in csv.DictReader(file)}

    # 准备本轮共用的检索器和模型；未传入索引时打开本地索引。
    if chroma_store is None:
        chroma_store = open_chroma_store()
    retriever = create_evidence_retriever(chroma_store)
    llm = create_llm(thinking=thinking).with_structured_output(method="json_mode")

    # 成本单独保存，沿用提交文件名，续跑时继续累计已有账目。
    cost_path = output_path.with_name(f"{output_path.stem}_cost.json")

    # 以追加模式写结果，只有空文件才写入表头。
    with output_path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        if file.tell() == 0:
            writer.writeheader()

        for number, question in enumerate(questions[start:], start + 1):
            question_id = question["id"]
            print(f"正在处理题目 {question_id}", flush=True)
            with record_cost(cost_path, question_id):
                prediction = predict_question(question, retriever, llm, paper_urls)

            # 成本保存成功后再写提交记录，避免续跑跳过尚未记录成本的题目。
            writer.writerow(prediction)
            file.flush()

            print(f"已保存 {number}/{len(questions)}", flush=True)

    print(f"已完成 {len(questions)} 道题，文件为 {output_path}")
    print(f"成本记录为 {cost_path}")




# 读取命令行参数，使用已有索引执行批量预测。
def main():
    args = parse_args()
    predict(args.questions, args.output, ids=args.ids, thinking=args.thinking)


if __name__ == "__main__":
    main()
