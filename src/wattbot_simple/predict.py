import argparse
import csv
import json
from pathlib import Path

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages.content import create_text_block

from wattbot_simple.cost import record_cost
from wattbot_simple.generate import build_evidence_blocks
from wattbot_simple.index import open_chroma_store
from wattbot_simple.model import create_llm
from wattbot_simple.retrieve import create_evidence_retriever


ROOT_DIR = Path(__file__).resolve().parents[2]
COLUMNS = (
    "id", "question", "answer", "answer_value", "answer_unit",
    "ref_id", "ref_url", "supporting_materials", "explanation",
)
SYSTEM_PROMPT = """只根据给定证据回答，并输出 JSON。
answer_value 只写直接答案，不写解释。
数值以输入的 answer_unit 为准；问题正文使用其他单位时，换算成指定单位。
按指定单位的数量级输出：原文为 2 million kWh，指定单位为 kWh 时写 "2000000"，为 million kWh 时写 "2"。
数值不附带单位，不使用千分位逗号或 million 等文字。
术语使用原文中最短且完整的名称，保留必要的修饰词和连接词，不为多个术语强制添加括号。
判断题用 "1" 表示真、"0" 表示假。
多个数值按题目要求的顺序、范围按下限到上限，写成 "(0.5,1.65)" 这样的字符串，不输出 JSON 数组。
citations 只填本次提供且实际支持答案的证据标签。
可回答时，citations 至少包含一个标签，supporting_materials 填原文摘录或图表定位。
explanation 简要说明证据如何支持答案。
证据不足时，将 answer_value 设为 "is_blank"，citations 设为空列表，
在 answer 和 explanation 中简要说明缺少什么证据。
除 citations 为字符串列表外，其余输出字段均为字符串。
answer_unit 为 "is_blank" 只表示没有指定单位，不表示无法回答。

JSON 示例
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
        raise OutputParserException("输出必须是 JSON 对象", llm_output=result_json)

    for field in ("answer", "answer_value", "supporting_materials", "explanation"):
        if field not in result or not isinstance(result[field], str):
            raise OutputParserException(
                f"{field} 缺失或不是字符串", llm_output=result_json
            )
        result[field] = result[field].strip()

    if "citations" not in result or not isinstance(result["citations"], list):
        raise OutputParserException(
            "citations 必须是字符串列表", llm_output=result_json
        )
    if any(
        not isinstance(label, str) or label not in citation_map
        for label in result["citations"]
    ):
        raise OutputParserException(
            "citations 含有无效的证据标签", llm_output=result_json
        )
    if not all(result[field] for field in ("answer", "answer_value", "explanation")):
        raise OutputParserException(
            "答案、答案值和解释不能为空", llm_output=result_json
        )
    if result["answer_value"] != "is_blank" and (
        not result["citations"] or not result["supporting_materials"]
    ):
        raise OutputParserException(
            "可回答的题目需要引用和佐证材料", llm_output=result_json
        )
    return result


# 完成一道比赛题的检索与回答，并整理成提交文件的一行。
def predict_question(question, retriever, llm, paper_urls):
    # 检索相关证据，并把题目指定单位一起传给模型。
    evidence = retriever.invoke(question["question"])
    evidence_blocks, citation_map = build_evidence_blocks(evidence)
    question_block = create_text_block(
        f"问题\n{question['question']}\n\n指定单位\n{question['answer_unit']}"
    )
    # 首次生成失败时，将具体输出错误反馈给模型，只纠正一次。
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content_blocks=[question_block, *evidence_blocks]),
    ]
    for attempt in range(2):
        try:
            result = llm.invoke(messages)
            result = normalize_prediction_result(result, citation_map)
            break
        except OutputParserException as exc:
            if attempt == 1:
                raise
            messages.extend([
                AIMessage(content=exc.llm_output or "上次未返回可解析的 JSON。"),
                HumanMessage(content=(
                    f"上次输出不符合要求：{exc}\n"
                    f"可用证据标签：{', '.join(citation_map)}。\n"
                    "请结合原问题和证据，重新输出完整 JSON。"
                    "仅引用实际支持答案的证据；证据不足时按要求返回 is_blank。"
                )),
            ])

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


# 读取题库路径、输出路径、指定题号和思考强度。
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


# 按题库顺序逐题预测并保存结果，重启时从已保存的位置继续。
def predict(questions_path, output_path, ids=None, chroma_store=None, thinking="none"):
    # 读取题库，并按指定题号选择本次处理范围。
    with questions_path.open(encoding="utf-8-sig", newline="") as file:
        questions = list(csv.DictReader(file))
    if ids:
        questions_by_id = {question["id"]: question for question in questions}
        questions = [questions_by_id[question_id] for question_id in ids]
    if not questions or len({q["id"] for q in questions}) != len(questions):
        raise ValueError("题库至少要有一道题，题号不能重复")

    # 先确认已有记录写入完整，再统计可跳过的题号。
    done_ids = read_completed_ids(output_path)

    # 完整记录必须对应本次题目顺序的连续前缀。
    if done_ids != [q["id"] for q in questions[:len(done_ids)]]:
        raise ValueError("已有输出与本次题目顺序不一致")

    # 从第一个尚未保存的题目继续，全部完成时直接结束。
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

        # 每题失败时带上题号向上传递异常，立即停止本轮预测。
        for number, question in enumerate(questions[start:], start + 1):
            try:
                with record_cost(cost_path, question["id"]):
                    prediction = predict_question(question, retriever, llm, paper_urls)

                # 成本保存成功后再写提交记录，避免续跑跳过尚未记录成本的题目。
                writer.writerow(prediction)
                file.flush()
            except Exception as exc:
                raise RuntimeError(f"题目 {question['id']} 处理失败") from exc

            print(f"已保存 {number}/{len(questions)}", flush=True)

    print(f"已完成 {len(questions)} 道题，文件为 {output_path}")
    print(f"成本记录为 {cost_path}")




# 读取命令行参数，使用已有索引执行批量预测。
def main():
    args = parse_args()
    predict(args.questions, args.output, ids=args.ids, thinking=args.thinking)


if __name__ == "__main__":
    main()
