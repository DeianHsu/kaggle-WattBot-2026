# 使用 Docling 解析 PDF 论文，并把正文、表格和图片相关文字切分为检索块。
# 保存结构化解析文档和图表原图，供回答时回查完整证据。
# 为每个文本块保留来源、原始元素和解析文件路径。

from pathlib import Path
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker import HybridChunker, DocChunk, DocMeta
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc import ImageRef, ImageRefMode, TextCellUnit, TextItem
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from wattbot_simple.model import describe_image, load_image_block

ROOT_DIR = Path(__file__).resolve().parents[2]


# 创建 PDF 转换器和分块器，供多篇论文共用。
def create_converter_and_chunker():
    # 保留 PDF 原始文字，并生成图片和裁剪表格需要的页面图像。
    pipeline_options = PdfPipelineOptions(
        do_ocr=False,
        generate_picture_images=True,
        generate_page_images=True,
        generate_parsed_pages=True,
        images_scale=2.0,
    )
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            )
        },
    )
    # 使用向量模型配套的分词器计算分块长度。
    tokenizer = HuggingFaceTokenizer.from_pretrained(
        model_name="BAAI/bge-small-en-v1.5",
        max_tokens=400,
    )
    chunker = HybridChunker(tokenizer=tokenizer)
    return converter, chunker


# 将解析文档转成检索块，并保留来源、原始元素和解析文件位置。
def chunk_pdf_document(pdf_path, pdf_document, document_path, chunker, parsed_pages, vlm):
    # 由分块器处理正文和表格，保留后续回查原文所需的元素标识。
    docling_chunks = list(chunker.chunk(pdf_document))

    # 图表补充文字共用分词器和长度限制，避免追加的文字块超长。
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        chunker.tokenizer.get_tokenizer(),
        chunk_size=chunker.tokenizer.get_max_tokens(),
        chunk_overlap=0,
    )

    # 图片优先按图题分块，缺少图题时用图内文字。
    for picture in pdf_document.pictures:
        picture_text = picture.caption_text(pdf_document).strip() or "\n".join(
            item.text
            for item, _ in pdf_document.iterate_items(
                root=picture, traverse_pictures=True
            )
            if isinstance(item, TextItem)
        ).strip()

        docling_chunks.extend(
            DocChunk(text=picture_text_chunk, meta=DocMeta(doc_items=[picture]))
            for picture_text_chunk in text_splitter.split_text(picture_text)
        )

    # 补取表格区域内的原始文字，减少表格解析遗漏对检索的影响。
    for table in pdf_document.tables:
        table_prov = table.prov[0]
        table_text = table.caption_text(pdf_document) + "\n" + " ".join(
            table_item.text
            for table_item in parsed_pages[table_prov.page_no].get_cells_in_bbox(
                cell_unit=TextCellUnit.WORD,
                bbox=table_prov.bbox,
            )
        )
        docling_chunks.extend(
            DocChunk(
                text=table_text_chunk,
                meta=DocMeta(doc_items=[table]),
            )
            for table_text_chunk in text_splitter.split_text(table_text)
        )

    # 每条描述只引用一张原图或表格截图，命中时沿用该元素的原图和页码。
    for item in [*pdf_document.pictures, *pdf_document.tables]:
        image_block = load_image_block(item.image, document_path.parent)
        description = describe_image(vlm, image_block)
        if not description:
            raise ValueError(f"{pdf_path.name} 的图表 {item.self_ref} 未生成描述")
        docling_chunks.append(DocChunk(text=description, meta=DocMeta(doc_items=[item])))

    # 过滤没有文字的块，再转换成检索和建库使用的文档对象。
    chunks = [
        Document(
            page_content=chunker.contextualize(chunk),
            metadata={
                "source": str(pdf_path),
                "document_path": str(document_path),
                "dl_meta": chunk.meta.export_json_dict(),
            },
        )
        for chunk in docling_chunks
        if chunk.text.strip()
    ]
    return chunks


# 解析一篇 PDF，保存原始解析产物并返回检索块。
def ingest_pdf(pdf_path, converter, chunker, vlm):
    # 同时保留解析文档和原始页面文字，供表格文字补充使用。
    pdf_result = converter.convert(pdf_path)
    pdf_document = pdf_result.document
    parsed_pages = {page.page_no: page.parsed_page for page in pdf_result.pages}

    # 将结构化文档和图片保存到论文对应目录，供问答时回查完整证据。
    output_dir = ROOT_DIR / "artifacts" / pdf_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    document_path = output_dir / "document.json"

    # 保存原表截图，并将相对路径写入表格元素，供回答时读取。
    for num, table in enumerate(pdf_document.tables):
        table_image = table.get_image(pdf_document)
        if table_image is None:
            raise ValueError(f"{pdf_path.name} 的第 {num + 1} 个表格无法生成原图")

        table_image_path = output_dir / f"table_{num}.png"
        table_image.save(table_image_path)
        table.image = ImageRef.from_pil(
            table_image,
            dpi=pdf_document.pages[table.prov[0].page_no].image.dpi,
        )
        table.image.uri = Path(table_image_path.name)

    # 图表已有独立图片，无需再将整页图像保存到解析产物中。
    for page in pdf_document.pages.values():
        page.image = None

    pdf_document.save_as_json(
        document_path,
        artifacts_dir=Path("document_artifacts"),
        image_mode=ImageRefMode.REFERENCED,
    )

    # 使用刚解析的文档生成检索块，避免重复解析 PDF。
    chunks = chunk_pdf_document(pdf_path, pdf_document, document_path, chunker, parsed_pages, vlm)
    return chunks
