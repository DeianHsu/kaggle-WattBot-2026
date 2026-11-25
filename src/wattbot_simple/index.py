# 建立和打开本地 Chroma 论文索引。

import json
from argparse import ArgumentParser
from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from wattbot_simple.model import create_llm

ROOT_DIR = Path(__file__).resolve().parents[2]
PAPERS_DIR = ROOT_DIR / "papers"


# 打开建库和检索共用的本地向量索引。
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


# 解析单篇未入库的论文并写入索引，已存在时跳过并返回 0。
def index_pdf(pdf_path, chroma_store, *, converter=None, chunker=None, vlm=None):
    from wattbot_simple.ingest import create_converter_and_chunker, ingest_pdf

    pdf_path = Path(pdf_path)
    pdf_stem = pdf_path.stem

    # 按论文文件名判断是否已入库，避免重复添加检索块。
    existing_ids = chroma_store.get(
        where={"pdf_stem": pdf_stem},
        limit=1,
        include=[],
    )["ids"]
    if existing_ids:
        return 0

    if vlm is None:
        vlm = create_llm()
    if converter is None or chunker is None:
        converter, chunker = create_converter_and_chunker()

    chunks = ingest_pdf(pdf_path, converter, chunker, vlm)
    if not chunks:
        raise ValueError(f"{pdf_path.name} 未生成可用的检索块")

    # 分配检索块标识，并将原始元素信息转成索引可保存的字符串。
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
    return len(chunks)


# 解析未入库的论文，将正文和图表的检索文本一并写入索引。
def build_chroma_index(papers_dir):
    from wattbot_simple.ingest import create_converter_and_chunker

    pdf_paths = sorted(papers_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"目录中没有 PDF：{papers_dir}")
    chroma_store = open_chroma_store()
    converter = chunker = vlm = None

    for paper_no, pdf_path in enumerate(pdf_paths, start=1):
        pdf_stem = pdf_path.stem

        # 先检查是否已入库，避免无需建库时初始化解析器和视觉模型。
        existing_ids = chroma_store.get(
            where={"pdf_stem": pdf_stem},
            limit=1,
            include=[],
        )["ids"]
        if existing_ids:
            print(f"已跳过 {pdf_path.name}", flush=True)
            continue

        # 处理首篇待入库论文时，初始化转换器、分块器和图表描述客户端。
        if converter is None:
            vlm = create_llm()
            converter, chunker = create_converter_and_chunker()

        index_pdf(
            pdf_path,
            chroma_store,
            converter=converter,
            chunker=chunker,
            vlm=vlm,
        )
        print(f"[{paper_no}/{len(pdf_paths)}] 已完成 {pdf_path.name}", flush=True)

    return chroma_store


# 直接运行本模块时，使用默认论文目录建库。
if __name__ == "__main__":
    ArgumentParser(description="解析论文并为图表生成检索描述").parse_args()
    build_chroma_index(PAPERS_DIR)
