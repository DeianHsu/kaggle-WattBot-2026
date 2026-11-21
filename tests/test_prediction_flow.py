import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import JsonOutputParser

from wattbot_simple import predict
from wattbot_simple.retrieve import retrieve_evidence


# 构造带固定用量的模型回执，便于核对多次调用的累计值。
def _response(content):
    return AIMessage(
        content=json.dumps(content),
        response_metadata={"model_name": "deepseek-flash"},
        usage_metadata={
            "input_tokens": 10,
            "output_tokens": 2,
            "total_tokens": 12,
            "input_token_details": {"cache_read": 3},
        },
    )


# 为每篇模拟论文提供一块带来源和页码的文本证据。
def _chunk(paper):
    return Document(
        page_content=f"{paper} reports a measurement.",
        metadata={
            "chunk_id": paper,
            "source": f"{paper}.pdf",
            "dl_meta": json.dumps({
                "doc_items": [{"label": "text", "prov": [{"page_no": 1}]}],
            }),
        },
    )


# 用模拟检索器和模型验证题目处理、引用和用量记录。
class PredictionFlowTest(unittest.TestCase):
    # 每个用例独立准备题库、论文链接和输出路径。
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="wattbot-prediction-")
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.questions = root / "questions.csv"
        self.questions.write_text(
            "id,question,answer_unit\n"
            "q1,Compare the two measurements,percent\n",
            encoding="utf-8",
        )
        (root / "input").mkdir()
        (root / "input" / "metadata.csv").write_text(
            "id,url\na,https://example.test/a\nb,https://example.test/b\n",
            encoding="utf-8",
        )
        self.output = root / "prediction.csv"
        self.cost = root / "prediction_cost.json"
        self.retriever = Mock()
        self.enterContext(patch.object(predict, "ROOT_DIR", root))
        self.enterContext(patch.object(
            predict, "create_evidence_retriever", return_value=self.retriever,
        ))
        self.create_llm = self.enterContext(
            patch.object(predict, "create_llm"),
        )

    # 按模型调用顺序安排回执，并沿用 JSON 解析流程。
    def _set_responses(self, *responses):
        model = FakeMessagesListChatModel(
            responses=[_response(response) for response in responses],
        )
        structured = JsonOutputParser()
        self.create_llm.return_value.with_structured_output.return_value = (
            model | structured
        )

    # 从批量入口运行用例，传入占位对象以跳过真实索引初始化。
    def _run(self):
        predict.predict(self.questions, self.output, chroma_store=object())

    # 读回磁盘账本，核对实际写入的单题记录和累计值。
    def _cost(self):
        return json.loads(self.cost.read_text(encoding="utf-8"))

    # 用五项事实检查规划出的子查询是否逐条执行。
    def test_retrieves_all_planned_fact_queries(self):
        question = "Compare five measurements"
        queries = [f"Measurement {number}" for number in range(1, 6)]
        chunks = [_chunk(query) for query in queries]
        self.retriever.invoke.side_effect = [[], *([chunk] for chunk in chunks)]
        llm = Mock()
        llm.invoke.return_value = {"queries": queries}
        evidence = retrieve_evidence(question, self.retriever, llm)

        self.assertEqual(
            self.retriever.invoke.call_args_list,
            [call(query) for query in [question, *queries]],
        )
        self.assertEqual(evidence, chunks)

    # 证据不足时，检查提交字段使用 is_blank，且检索停在首轮。
    def test_missing_evidence_returns_blank_without_more_search(self):
        self.retriever.invoke.return_value = [_chunk("a")]
        self._set_responses(
            {"queries": []},
            {
                "answer": "Evidence is incomplete.",
                "answer_value": "is_blank",
                "explanation": "The second measurement is missing.",
                "supporting_materials": "",
                "citations": [],
                "missing_queries": ["Second measurement"],
            },
        )

        self._run()

        self.retriever.invoke.assert_called_once_with(
            "Compare the two measurements",
        )
        with self.output.open(encoding="utf-8", newline="") as file:
            row, = csv.DictReader(file)
        for field in ("answer_value", "ref_id", "ref_url", "supporting_materials"):
            self.assertEqual(row[field], "is_blank")
        self.assertEqual(self._cost()["total"]["total_tokens"], 24)

    # 用跨论文回答检查引用还原以及规划、作答两次调用的用量。
    def test_citations_and_model_usage_are_saved(self):
        self.retriever.invoke.side_effect = [
            [_chunk("a")], [_chunk("a")], [_chunk("b")],
        ]
        self._set_responses(
            {"queries": ["First measurement", "Second measurement"]},
            {
                "answer": "12 percent",
                "answer_value": "12",
                "explanation": "Both measurements are available.",
                "supporting_materials": "Both measurements are in the evidence.",
                "citations": ["Evidence_0", "Evidence_1"],
            },
        )

        self._run()

        self.assertEqual(
            self.retriever.invoke.call_args_list,
            [call("Compare the two measurements"), call("First measurement"),
             call("Second measurement")],
        )
        with self.output.open(encoding="utf-8", newline="") as file:
            row, = csv.DictReader(file)
        self.assertEqual(row["answer_value"], "12")
        self.assertEqual(json.loads(row["ref_id"]), ["a", "b"])
        record, = self._cost()["records"]
        self.assertEqual(record["status"], "success")
        self.assertEqual(record["total_tokens"], 24)
        self.assertEqual(record["cache_hit_tokens"], 6)
        self.assertEqual(self._cost()["total"]["total_tokens"], 24)

    # 验证输出校验失败后保留用量，重跑时从失败的题目继续。
    def test_failed_prediction_preserves_usage_and_can_resume(self):
        self.retriever.invoke.return_value = [_chunk("a")]
        self._set_responses({"queries": []}, {}, {})

        with self.assertRaises(OutputParserException):
            self._run()

        self.assertEqual(predict.read_completed_ids(self.output), [])
        record, = self._cost()["records"]
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["total_tokens"], 36)

        self._set_responses(
            {"queries": []},
            {
                "answer": "12 percent",
                "answer_value": "12",
                "explanation": "The source reports 12 percent.",
                "supporting_materials": "12 percent",
                "citations": ["Evidence_0"],
            },
        )
        self._run()
        self._run()

        self.assertEqual(predict.read_completed_ids(self.output), ["q1"])
        cost = self._cost()
        self.assertEqual(
            [record["status"] for record in cost["records"]],
            ["failed", "success"],
        )
        self.assertEqual(cost["total"]["total_tokens"], 60)
