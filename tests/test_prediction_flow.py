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
from wattbot_simple.generate import generate_with_retrieval


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


class PredictionFlowTest(unittest.TestCase):
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

    def _set_responses(self, *responses):
        model = FakeMessagesListChatModel(
            responses=[_response(response) for response in responses],
        )
        structured = JsonOutputParser()
        self.create_llm.return_value.with_structured_output.return_value = (
            model | structured
        )

    def _run(self):
        predict.predict(self.questions, self.output, chroma_store=object())

    def _cost(self):
        return json.loads(self.cost.read_text(encoding="utf-8"))

    def test_retrieves_all_planned_fact_queries(self):
        question = "Compare five measurements"
        queries = [f"Measurement {number}" for number in range(1, 6)]
        chunks = [_chunk(query) for query in queries]
        self.retriever.invoke.side_effect = [[], *([chunk] for chunk in chunks)]
        llm = Mock()
        llm.invoke.return_value = {"queries": queries}
        draft_answer = Mock(return_value=({}, {}))

        generate_with_retrieval(question, self.retriever, llm, draft_answer)

        self.assertEqual(
            self.retriever.invoke.call_args_list,
            [call(query) for query in [question, *queries]],
        )
        draft_answer.assert_called_once_with(chunks)

    def test_follow_up_citations_and_all_model_usage_are_saved(self):
        self.retriever.invoke.side_effect = [
            [_chunk("a")], [_chunk("a")], [_chunk("b")],
        ]
        self._set_responses(
            {"queries": ["First measurement"]},
            {
                "answer": "Evidence is incomplete.",
                "answer_value": "is_blank",
                "explanation": "The second measurement is missing.",
                "supporting_materials": "",
                "citations": [],
                "missing_queries": ["Second measurement"],
            },
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
        self.assertEqual(record["total_tokens"], 36)
        self.assertEqual(record["cache_hit_tokens"], 9)
        self.assertEqual(self._cost()["total"]["total_tokens"], 36)

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
