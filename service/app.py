import base64
import binascii
import csv
from contextlib import asynccontextmanager
from pathlib import Path
import threading
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, StringConstraints

from wattbot_simple.generate import generate_answer
from wattbot_simple.index import index_pdf, open_chroma_store
from wattbot_simple.model import create_llm
from wattbot_simple.retrieve import (
    create_evidence_retriever,
    create_reranker,
    retrieve_evidence,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
PAPERS_DIR = ROOT_DIR / "papers"
METADATA_PATH = ROOT_DIR / "input" / "metadata.csv"


# 接收一道题并校验题目，调用方可以指定答案单位。
class AnswerRequest(BaseModel):
    question: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1)
    ]
    answer_unit: str | None = None


# 记录一篇被引用论文的编号、链接和证据页码。
class Citation(BaseModel):
    paper_id: str
    url: str
    pages: list[int]


# 定义接口返回给调用方的答案和论文引用。
class AnswerResponse(BaseModel):
    answer: str
    citations: list[Citation]


# 接收待增量入库的单篇 PDF（Base64 编码）和可选来源链接。
class IngestRequest(BaseModel):
    filename: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1)
    ]
    content_base64: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1)
    ]
    url: str = ""


# 返回单篇 PDF 的入库状态与新增检索块数量。
class IngestResponse(BaseModel):
    paper_id: str
    status: str
    chunks_added: int


# 启动时加载论文链接表，缺少元数据文件时回退为空映射。
def _load_paper_urls(metadata_path: Path) -> dict[str, str]:
    if not metadata_path.is_file():
        return {}
    with metadata_path.open(encoding="utf-8-sig", newline="") as file:
        return {row["id"]: row["url"] for row in csv.DictReader(file)}


# 启动时准备论文链接、检索器、重排器和模型客户端，供后续请求共用。
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.index_lock = threading.RLock()
    app.state.paper_urls = _load_paper_urls(METADATA_PATH)
    app.state.chroma_store = open_chroma_store()
    app.state.reranker = create_reranker()
    app.state.retriever = create_evidence_retriever(
        app.state.chroma_store, reranker=app.state.reranker,
    )
    app.state.llm = create_llm().with_structured_output(method="json_mode")
    yield


app = FastAPI(title="WattBot Simple", lifespan=lifespan)


# 返回服务的本地就绪状态，供调用方检查。
@app.get("/health")
def health() -> dict[str, str]:
    # 服务在本地资源初始化完成后才接收请求。
    return {"status": "ok"}


# 检索并回答一道题，再将引用的证据标签还原为论文链接和 PDF 页码。
@app.post("/answer", operation_id="wattbot_answer")
def answer(body: AnswerRequest, request: Request) -> AnswerResponse:
    state = request.app.state
    answer_unit = (body.answer_unit or "").strip() or "is_blank"
    # 短暂加锁读取当前检索器与链接映射快照，不阻塞耗时模型调用。
    with state.index_lock:
        retriever = state.retriever
        paper_urls = dict(state.paper_urls)

    evidence = retrieve_evidence(
        body.question, retriever, state.llm, answer_unit=answer_unit,
    )
    result, citation_map = generate_answer(
        body.question,
        evidence,
        state.llm,
        answer_unit,
        follow_question_language=True,
    )
    if result["answer_value"] == "is_blank":
        return AnswerResponse(answer=result["answer"], citations=[])

    # 同一篇论文的多个证据块可能覆盖相同或不同的 PDF 页。
    pages_by_paper: dict[str, set[int]] = {}
    for label in result["citations"]:
        citation = citation_map[label]
        paper_id = Path(citation["source"]).stem
        pages_by_paper.setdefault(paper_id, set()).update(citation["pages"])

    citations = [
        Citation(
            paper_id=paper_id,
            url=paper_urls.get(paper_id, ""),
            pages=sorted(pages),
        )
        for paper_id, pages in pages_by_paper.items()
    ]
    return AnswerResponse(answer=result["answer"], citations=citations)


# 接收单篇 PDF 增量建库，并复用重排器热刷新内存中的混合检索索引。
@app.post("/ingest", operation_id="wattbot_ingest")
def ingest(body: IngestRequest, request: Request) -> IngestResponse:
    safe_name = Path(body.filename).name
    if Path(safe_name).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="仅支持上传 .pdf 文件")

    try:
        pdf_bytes = base64.b64decode(body.content_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="content_base64 不是有效的 Base64 数据") from exc

    if not pdf_bytes.startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="上传内容不是有效的 PDF 文件")

    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = PAPERS_DIR / safe_name
    paper_id = pdf_path.stem
    state = request.app.state

    with state.index_lock:
        if not pdf_path.exists():
            pdf_path.write_bytes(pdf_bytes)

        chunks_added = index_pdf(pdf_path, state.chroma_store)
        if body.url.strip():
            state.paper_urls[paper_id] = body.url.strip()

        if chunks_added == 0:
            return IngestResponse(
                paper_id=paper_id, status="skipped", chunks_added=0,
            )

        state.retriever = create_evidence_retriever(
            state.chroma_store, reranker=state.reranker,
        )

    return IngestResponse(
        paper_id=paper_id, status="indexed", chunks_added=chunks_added,
    )
