import json
from argparse import ArgumentParser
from pathlib import Path
from docling_core.transforms.chunker import DocMeta
from docling_core.types import DoclingDocument
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from wattbot_simple.ingest import create_converter_and_chunker, ingest_pdf, parsed_document_path
from wattbot_simple.model import create_picture_vlm, describe_picture

ROOT_DIR = Path(__file__).resolve().parents[2]
PAPERS_DIR = ROOT_DIR / "papers"


# 使用 BGE 向量模型打开本地 Chroma 索引。
def open_chroma_store():
    # 使用归一化向量，使建库与查询采用相同的向量表示。
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


# 每张图片独立写入。续跑时按原始元素标识跳过已保存的描述。
def index_picture_descriptions(chroma_store, pdf_path, existing_ids, vlm):
    pdf_stem = pdf_path.stem
    document_path = parsed_document_path(pdf_path)
    pdf_document = DoclingDocument.load_from_json(document_path)
    added = 0
    for picture in pdf_document.pictures:
        chunk_id = f"{pdf_stem}:picture_description:{picture.self_ref}"
        if chunk_id in existing_ids:
            continue

        try:
            if picture.image is None:
                raise ValueError("解析文档缺少原图引用")
            image_path = document_path.parent / str(picture.image.uri)
            description = describe_picture(vlm, image_path, picture.image.mimetype)
            doc_meta = DocMeta(doc_items=[picture]).export_json_dict()
            chunk = Document(
                page_content=description,
                metadata={
                    "source": str(pdf_path),
                    "document_path": str(document_path),
                    "chunk_id": chunk_id,
                    "pdf_stem": pdf_stem,
                    "picture_description_model": vlm.model_name,
                    "dl_meta": json.dumps(doc_meta, ensure_ascii=False),
                },
            )
            chroma_store.add_documents(documents=[chunk], ids=[chunk_id])
        except Exception as exc:
            raise RuntimeError(f"{pdf_path.name} 的图片 {picture.self_ref} 描述入库失败") from exc
        added += 1
    return added


# 为尚未入库的论文建立索引，并按需为已有解析图片补充描述。
def build_chroma_index(papers_dir, *, describe_pictures=False, paper_id=None):
    # 收集论文，并打开索引检查已有记录。
    pdf_paths = sorted(papers_dir.glob("*.pdf"))
    if paper_id is not None:
        pdf_paths = [pdf_path for pdf_path in pdf_paths if pdf_path.stem == paper_id]
        if not pdf_paths:
            raise FileNotFoundError(f"目录 {papers_dir} 中没有论文 {paper_id}.pdf")
    if not pdf_paths:
        raise FileNotFoundError(f"目录中没有 PDF：{papers_dir}")
    vlm = create_picture_vlm() if describe_pictures else None
    chroma_store = open_chroma_store()
    converter = chunker = None

    for paper_no, pdf_path in enumerate(pdf_paths, start=1):
        pdf_stem = pdf_path.stem

        # 补充图片描述时读取已有块标识，避免重复调用视觉模型。
        existing_ids = set(chroma_store.get(
            where={"pdf_stem": pdf_stem},
            limit=1 if vlm is None else None,
            include=[],
        )["ids"])
        if existing_ids and vlm is None:
            print(f"已跳过 {pdf_path.name}", flush=True)
            continue

        chunks = []
        if not existing_ids:
            # 首次遇到未入库论文时创建解析组件，后续论文共用。
            if converter is None:
                converter, chunker = create_converter_and_chunker()

            # 解析当前论文，保存原图并准备正文、图题和表格块。
            chunks = ingest_pdf(pdf_path, converter, chunker)

            # 为每个块设置唯一标识，并把解析元数据转成可存储的字符串。
            for chunk_no, chunk in enumerate(chunks):
                chunk_id = f"{pdf_stem}_{chunk_no}"
                chunk.metadata.update(
                    chunk_id=chunk_id,
                    pdf_stem=pdf_stem,
                    dl_meta=json.dumps(
                        chunk.metadata["dl_meta"],
                        ensure_ascii=False,
                    )
                )

            if chunks:
                chroma_store.add_documents(
                    documents=chunks,
                    ids=[chunk.metadata["chunk_id"] for chunk in chunks],
                )

        progress = f"[{paper_no}/{len(pdf_paths)}] 已完成 {pdf_path.name}"
        added = 0
        if vlm is not None:
            added = index_picture_descriptions(chroma_store, pdf_path, existing_ids, vlm)
            progress += f"，新增 {added} 条图片描述"
        if not (existing_ids or chunks or added):
            raise ValueError(f"{pdf_path.name} 未生成可用的检索块")
        print(progress, flush=True)

    return chroma_store


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--describe-pictures", action="store_true", help="将视觉语言模型生成的图片描述写入索引",
    )
    parser.add_argument("--paper", help="只处理指定论文，传入 PDF 文件名主干")
    args = parser.parse_args()
    build_chroma_index(PAPERS_DIR, describe_pictures=args.describe_pictures, paper_id=args.paper)
