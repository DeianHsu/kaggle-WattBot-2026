# 建立和打开本地 Chroma 向量索引和 BM25 关键词索引。

import gzip
import json
import re
from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

from wattbot_simple.ingest import create_converter_and_chunker, ingest_pdf
from wattbot_simple.model import create_llm

ROOT_DIR = Path(__file__).resolve().parents[2]
PAPERS_DIR = ROOT_DIR / "papers"


def open_chroma_store():
    # 建库和查询共用向量模型与归一化设置。
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={"normalize_embeddings": True},
    )
    # 打开固定的持久化集合；集合不存在时创建它。
    chroma_store = Chroma(
        collection_name="wattbot_text",
        embedding_function=embedding_model,
        persist_directory=str(ROOT_DIR / "chroma_db"),
    )
    return chroma_store


def _tokenize_bm25(text):
    return re.findall(r"\w+", text.lower())


def _bm25_path(chroma_store):
    settings = chroma_store._client.get_settings()
    if settings.is_persistent:
        return Path(settings.persist_directory) / "bm25.json.gz"


# 使用 Chroma 中的全部检索块构建 BM25，并覆盖磁盘上的旧索引。
def build_bm25_index(chroma_store):
    chunk_ids = chroma_store.get(include=[])["ids"]
    if not chunk_ids:
        raise ValueError("索引为空，请先完成建库")
    chunks = chroma_store.get_by_ids(chunk_ids)
    retriever = BM25Retriever.from_documents(
        documents=chunks,
        k=30,
        preprocess_func=_tokenize_bm25,
    )

    index_path = _bm25_path(chroma_store)
    if index_path is None:
        return retriever

    snapshot = {
        "collection_id": str(chroma_store._collection.id),
        "documents": [doc.model_dump() for doc in retriever.docs],
        "vectorizer": vars(retriever.vectorizer),
    }
    # 写完再替换，避免预测读到只写了一部分的索引。
    with TemporaryDirectory(dir=index_path.parent) as directory:
        temporary_path = Path(directory) / index_path.name
        with gzip.open(
            temporary_path, "wt", compresslevel=1, encoding="utf-8",
        ) as file:
            json.dump(snapshot, file, ensure_ascii=False, separators=(",", ":"))
        temporary_path.replace(index_path)
    return retriever


# 预测时读取已保存的 BM25。旧库缺文件或检索块发生变化时补建。
def open_bm25_retriever(chroma_store):
    index_path = _bm25_path(chroma_store)
    if index_path is None:
        return build_bm25_index(chroma_store)
    try:
        with gzip.open(index_path, "rt", encoding="utf-8") as file:
            snapshot = json.load(file)
    except FileNotFoundError:
        return build_bm25_index(chroma_store)

    documents = snapshot["documents"]
    chunk_ids = chroma_store.get(include=[])["ids"]
    if (
        snapshot["collection_id"] != str(chroma_store._collection.id)
        or {doc["id"] for doc in documents} != set(chunk_ids)
    ):
        return build_bm25_index(chroma_store)

    # 按落盘顺序恢复文档和词频，保留 BM25 同分时的排序。
    vectorizer = BM25Okapi.__new__(BM25Okapi)
    vars(vectorizer).update(snapshot["vectorizer"])
    return BM25Retriever(
        vectorizer=vectorizer,
        docs=[Document.model_validate(doc) for doc in documents],
        k=30,
        preprocess_func=_tokenize_bm25,
    )


# 解析未入库的论文，将正文和图表的检索文本一并写入索引。
def build_chroma_index(papers_dir):
    pdf_paths = sorted(papers_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"目录中没有 PDF：{papers_dir}")
    chroma_store = open_chroma_store()
    converter = chunker = vlm = None

    for paper_no, pdf_path in enumerate(pdf_paths, start=1):
        pdf_stem = pdf_path.stem

        existing_ids = chroma_store.get(
            where={"pdf_stem": pdf_stem},
            limit=1,
            include=[],
        )["ids"]
        if existing_ids:
            print(f"已跳过 {pdf_path.name}", flush=True)
            continue

        if converter is None:
            vlm = create_llm()
            converter, chunker = create_converter_and_chunker()

        chunks = ingest_pdf(pdf_path, converter, chunker, vlm)
        if not chunks:
            raise ValueError(f"{pdf_path.name} 未生成可用的检索块")

        for chunk_no, chunk in enumerate(chunks):
            chunk_id = f"{pdf_stem}_{chunk_no}"
            chunk.metadata.update(
                chunk_id=chunk_id,
                pdf_stem=pdf_stem,
                dl_meta=json.dumps(chunk.metadata["dl_meta"], ensure_ascii=False),
            )

        chroma_store.add_documents(
            documents=chunks,
            ids=[chunk.metadata["chunk_id"] for chunk in chunks],
        )
        print(f"[{paper_no}/{len(pdf_paths)}] 已完成 {pdf_path.name}", flush=True)

    build_bm25_index(chroma_store)
    return chroma_store


if __name__ == "__main__":
    ArgumentParser(description="解析论文并为图表生成检索描述").parse_args()
    build_chroma_index(PAPERS_DIR)
