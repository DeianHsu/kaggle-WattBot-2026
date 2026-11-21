# 批量生成比赛提交文件，按题保存答案和成本，支持从已有结果续跑。

import argparse
import csv
import json
from pathlib import Path

from wattbot_simple.cost import record_cost
from wattbot_simple.generate import generate_answer
from wattbot_simple.index import open_chroma_store
from wattbot_simple.model import create_llm
from wattbot_simple.retrieve import create_evidence_retriever


ROOT_DIR = Path(__file__).resolve().parents[2]
COLUMNS = (
    "id", "question", "answer", "answer_value", "answer_unit",
    "ref_id", "ref_url", "supporting_materials", "explanation",
)

# 将一道题的答案和引用整理成提交文件的一行。
def predict_question(question, retriever, llm, paper_urls):
    result, citation_map = generate_answer(
        question["question"], retriever, llm,
        answer_unit=question["answer_unit"],
    )
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
