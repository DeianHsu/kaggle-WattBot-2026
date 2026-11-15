import json
from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from wattbot_simple.ingest import ingest_pdf, create_converter_and_chunker

ROOT_DIR = Path(__file__).resolve().parents[2]
PAPERS_DIR = ROOT_DIR / "papers"


# 使用 BGE 向量模型打开本地 Chroma 索引。
def open_chroma_store():
    # 使用归一化向量，使建库与查询采用相同的向量表示。
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={"normalize_embeddings": True},
    )
    # 打开固定的持久化集合；集合不存在时创建它。
    chroma_store = Chroma(
        collection_name="wattbot_text",
        embedding_function=embeddings,
        persist_directory=str(ROOT_DIR / "chroma_db"),
    )
    return chroma_store


# 扫描论文目录，将尚未入库的论文解析后写入索引。
def build_chroma_index(papers_dir):
    # 收集论文，并为整轮建库准备可复用的转换器、分块器和索引。
    pdf_paths = sorted(papers_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"目录中没有 PDF：{papers_dir}")
    converter, chunker = create_converter_and_chunker()
    chroma_store = open_chroma_store()

    for number, pdf_path in enumerate(pdf_paths, start=1):
        pdf_stem = pdf_path.stem

        # 已有分块的论文直接跳过，避免重复解析和写入。
        existing_ids = chroma_store.get(
            where={"pdf_stem": pdf_stem},
            limit=1,
            include=[],
        )["ids"]
        if existing_ids:
            print(f"已跳过 {pdf_path.name}", flush=True)
            continue

        # 解析当前论文，确认产生了可写入索引的文本块。
        chunks = ingest_pdf(pdf_path, converter, chunker)
        if not chunks:
            raise ValueError(f"{pdf_path.name} 未生成可用的检索块")

        # 为每个块设置唯一标识，并把解析元数据转成可存储的字符串。
        for num, chunk in enumerate(chunks):
            chunk_id = f"{pdf_stem}_{num}"
            chunk.metadata.update(
                chunk_id=chunk_id,
                pdf_stem=pdf_stem,
                dl_meta=json.dumps(
                    chunk.metadata["dl_meta"],
                    ensure_ascii=False,
                )
            )

        # 批量计算向量并写入索引，随后显示当前进度。
        chroma_store.add_documents(
            documents=chunks,
            ids=[chunk.metadata["chunk_id"] for chunk in chunks],
        )

        print(f"[{number}/{len(pdf_paths)}] 已完成 {pdf_path.name}", flush=True)

    return chroma_store


if __name__ == "__main__":
    build_chroma_index(PAPERS_DIR)
