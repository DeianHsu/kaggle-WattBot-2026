import base64
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from service import app as api_module


# 使用离线模拟组件验证 FastAPI 接口的问答、容错与增量入库流程。
class ServiceApiTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="wattbot-api-")
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.papers_dir = root / "papers"
        self.metadata_path = root / "input" / "metadata.csv"
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.write_text(
            "id,url\niea2025,https://example.test/iea2025.pdf\n",
            encoding="utf-8",
        )

        self.chroma_store = Mock(name="chroma_store")
        self.reranker = Mock(name="reranker")
        self.initial_retriever = Mock(name="initial_retriever")
        self.refreshed_retriever = Mock(name="refreshed_retriever")
        self.llm = Mock(name="llm")

        self.enterContext(patch.object(api_module, "PAPERS_DIR", self.papers_dir))
        self.enterContext(
            patch.object(api_module, "METADATA_PATH", self.metadata_path)
        )
        self.enterContext(
            patch.object(
                api_module, "open_chroma_store", return_value=self.chroma_store,
            )
        )
        self.enterContext(
            patch.object(api_module, "create_reranker", return_value=self.reranker)
        )
        self.create_retriever = self.enterContext(
            patch.object(
                api_module,
                "create_evidence_retriever",
                side_effect=[self.initial_retriever, self.refreshed_retriever],
            )
        )
        create_llm = self.enterContext(patch.object(api_module, "create_llm"))
        create_llm.return_value.with_structured_output.return_value = self.llm

        self.client = self.enterContext(TestClient(api_module.app))

    # 验证健康检查接口返回就绪状态。
    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    # 验证问答接口按问题语言作答，并合并去重同一篇论文的多页引用。
    def test_answer_merges_pages_and_uses_question_language(self):
        with (
            patch.object(
                api_module, "retrieve_evidence", return_value=["chunk_1", "chunk_2"],
            ) as mock_retrieve,
            patch.object(
                api_module,
                "generate_answer",
                return_value=(
                    {
                        "answer": "2024 年数据中心约占全球用电量的 1.5%。",
                        "answer_value": "1.5",
                        "citations": ["Evidence_0", "Evidence_1"],
                    },
                    {
                        "Evidence_0": {
                            "source": "papers/iea2025.pdf",
                            "pages": [14, 13],
                        },
                        "Evidence_1": {
                            "source": "papers/iea2025.pdf",
                            "pages": [14, 49],
                        },
                    },
                ),
            ) as mock_generate,
        ):
            response = self.client.post(
                "/answer",
                json={"question": "2024年数据中心占全球用电量多少？", "answer_unit": "percent"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "answer": "2024 年数据中心约占全球用电量的 1.5%。",
                "citations": [
                    {
                        "paper_id": "iea2025",
                        "url": "https://example.test/iea2025.pdf",
                        "pages": [13, 14, 49],
                    }
                ],
            },
        )
        mock_retrieve.assert_called_once_with(
            "2024年数据中心占全球用电量多少？",
            self.initial_retriever,
            self.llm,
            answer_unit="percent",
        )
        mock_generate.assert_called_once_with(
            "2024年数据中心占全球用电量多少？",
            ["chunk_1", "chunk_2"],
            self.llm,
            "percent",
            follow_question_language=True,
        )

    # 验证未登记在元数据表中的新论文也能安全返回引用，不抛 KeyError。
    def test_answer_handles_missing_metadata_url(self):
        with (
            patch.object(api_module, "retrieve_evidence", return_value=["chunk_1"]),
            patch.object(
                api_module,
                "generate_answer",
                return_value=(
                    {
                        "answer": "Found in uploaded paper.",
                        "answer_value": "42",
                        "citations": ["Evidence_0"],
                    },
                    {
                        "Evidence_0": {
                            "source": "papers/custom_report.pdf",
                            "pages": [3],
                        },
                    },
                ),
            ),
        ):
            response = self.client.post("/answer", json={"question": "What is the value?"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["citations"],
            [{"paper_id": "custom_report", "url": "", "pages": [3]}],
        )

    # 验证空白输入被校验拦截，证据不足时返回空引用列表。
    def test_answer_validation_and_insufficient_evidence(self):
        blank_response = self.client.post("/answer", json={"question": "   "})
        self.assertEqual(blank_response.status_code, 422)

        with (
            patch.object(api_module, "retrieve_evidence", return_value=[]),
            patch.object(
                api_module,
                "generate_answer",
                return_value=(
                    {
                        "answer": "证据不足，无法回答该问题。",
                        "answer_value": "is_blank",
                        "citations": [],
                    },
                    {},
                ),
            ),
        ):
            missing_response = self.client.post(
                "/answer", json={"question": "未知年份的数据是多少？"},
            )

        self.assertEqual(missing_response.status_code, 200)
        self.assertEqual(
            missing_response.json(),
            {"answer": "证据不足，无法回答该问题。", "citations": []},
        )

    # 验证增量入库会落盘 PDF、更新链接并复用重排器热刷新检索器。
    def test_ingest_indexes_pdf_and_refreshes_retriever(self):
        pdf_payload = base64.b64encode(b"%PDF-1.4\n%mock pdf\n").decode("ascii")
        with patch.object(api_module, "index_pdf", return_value=5) as mock_index:
            response = self.client.post(
                "/ingest",
                json={
                    "filename": "../nested/new_paper.pdf",
                    "content_base64": pdf_payload,
                    "url": "https://example.test/new_paper.pdf",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"paper_id": "new_paper", "status": "indexed", "chunks_added": 5},
        )
        saved_pdf = self.papers_dir / "new_paper.pdf"
        self.assertTrue(saved_pdf.is_file())
        mock_index.assert_called_once_with(saved_pdf, self.chroma_store)
        self.assertEqual(
            api_module.app.state.paper_urls["new_paper"],
            "https://example.test/new_paper.pdf",
        )
        self.assertIs(api_module.app.state.retriever, self.refreshed_retriever)
        self.assertEqual(self.create_retriever.call_count, 2)

    # 验证非 PDF、无效内容被拒绝，以及已存在文档跳过重建检索器。
    def test_ingest_validation_and_skip_existing(self):
        non_pdf = self.client.post(
            "/ingest",
            json={"filename": "notes.txt", "content_base64": "aGVsbG8="},
        )
        self.assertEqual(non_pdf.status_code, 400)

        bad_header = self.client.post(
            "/ingest",
            json={
                "filename": "fake.pdf",
                "content_base64": base64.b64encode(b"not a pdf").decode("ascii"),
            },
        )
        self.assertEqual(bad_header.status_code, 400)

        valid_pdf = base64.b64encode(b"%PDF-1.7\n").decode("ascii")
        with patch.object(api_module, "index_pdf", return_value=0):
            skipped = self.client.post(
                "/ingest",
                json={"filename": "iea2025.pdf", "content_base64": valid_pdf},
            )

        self.assertEqual(skipped.status_code, 200)
        self.assertEqual(
            skipped.json(),
            {"paper_id": "iea2025", "status": "skipped", "chunks_added": 0},
        )
        self.assertEqual(self.create_retriever.call_count, 1)

    # 验证多语言提示词切换与 Docling 解析文档缓存生效。
    def test_prompt_language_and_docling_cache(self):
        from wattbot_simple import generate

        default_prompt = generate._build_answer_prompt(
            follow_question_language=False,
        )
        localized_prompt = generate._build_answer_prompt(
            follow_question_language=True,
        )
        self.assertIn("Write answer and explanation in English.", default_prompt)
        self.assertIn("same language as the question", localized_prompt)

        generate._load_docling_document.cache_clear()
        with patch.object(
            generate.DoclingDocument, "load_from_json", return_value=object(),
        ) as mock_load:
            first = generate._load_docling_document("artifacts/iea2025/document.json")
            second = generate._load_docling_document("artifacts/iea2025/document.json")
        self.assertIs(first, second)
        mock_load.assert_called_once()


if __name__ == "__main__":
    unittest.main()
