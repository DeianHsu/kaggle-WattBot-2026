import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from wattbot_simple import index


class BM25IndexTest(unittest.TestCase):
    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(directory)
        self.index_path = self.root / "bm25.json.gz"
        self.chroma_store = Mock()
        self.chroma_store._client.get_settings.return_value = SimpleNamespace(
            is_persistent=True, persist_directory=directory,
        )
        self.chroma_store._collection.id = "test-collection"
        self.documents = [
            Document(
                id=f"paper_{number}",
                page_content=text,
                metadata={"chunk_id": f"paper_{number}", "pdf_stem": "paper"},
            )
            for number, text in enumerate([
                "Solar output was 12 kWh.",
                "The panel produced solar power.",
                "The facility used water for cooling.",
            ])
        ]
        self.chroma_store.get.side_effect = self._get_ids
        # Chroma 返回的文档顺序可以与请求的编号顺序不同。
        self.chroma_store.get_by_ids.side_effect = lambda ids: self.documents[::-1]

    def _get_ids(self, include, where=None, limit=None):
        ids = [
            doc.id
            for doc in self.documents
            if where is None or where["pdf_stem"] == doc.metadata["pdf_stem"]
        ]
        return {"ids": ids[:limit]}

    def test_existing_chroma_is_backfilled_and_reload_keeps_rank_order(self):
        expected = BM25Retriever.from_documents(
            documents=self.documents[::-1],
            k=30,
            preprocess_func=lambda text: re.findall(r"\w+", text.lower()),
        )

        index.open_bm25_retriever(self.chroma_store)
        self.assertTrue(self.index_path.is_file())
        self.chroma_store.get_by_ids.reset_mock()
        reopened = index.open_bm25_retriever(self.chroma_store)
        self.chroma_store.get_by_ids.assert_not_called()

        for query in ("SOLAR / kWh", "unknown term"):
            with self.subTest(query=query):
                self.assertEqual(reopened.invoke(query), expected.invoke(query))

    def test_build_index_refreshes_bm25_after_adding_a_paper(self):
        index.open_bm25_retriever(self.chroma_store)
        (self.root / "paper.pdf").touch()
        (self.root / "new.pdf").touch()
        chunk = Document(
            page_content="Wind output was 30 kWh.",
            metadata={"source": "new.pdf", "dl_meta": {"doc_items": []}},
        )

        def add_documents(documents, ids):
            self.documents.extend(
                document.model_copy(update={"id": doc_id})
                for document, doc_id in zip(documents, ids)
            )

        self.chroma_store.add_documents.side_effect = add_documents
        with (
            patch.object(index, "open_chroma_store", return_value=self.chroma_store),
            patch.object(index, "create_llm"),
            patch.object(index, "create_converter_and_chunker", return_value=(1, 2)),
            patch.object(index, "ingest_pdf", return_value=[chunk]),
        ):
            self.assertIs(index.build_chroma_index(self.root), self.chroma_store)

        self.chroma_store.get_by_ids.reset_mock()
        reopened = index.open_bm25_retriever(self.chroma_store)
        self.chroma_store.get_by_ids.assert_not_called()
        self.assertEqual(reopened.invoke("wind")[0].id, "new_0")
