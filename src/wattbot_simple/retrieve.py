import re
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever


# 组合语义检索和关键词检索，用同一批文本块提供候选证据。
def create_hybrid_retriever(chroma_store):
    # 读取全部索引文本，供关键词检索器建立词频统计。
    chunk_ids = chroma_store.get(include=[])["ids"]
    if not chunk_ids:
        raise ValueError("索引为空，请先完成建库")
    chunks = chroma_store.get_by_ids(chunk_ids)
    # 按问题与文本的语义相似度召回十二个候选块。
    vector_retriever = chroma_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 30},
    )
    # 将文字统一为小写词项，再按关键词相关性召回十二个候选块。
    bm25_retriever = BM25Retriever.from_documents(
        documents=chunks,
        k=30,
        preprocess_func=lambda text: re.findall(r"\w+", text.lower()),
    )
    # 合并两路候选排名，并按文本块标识去重。
    return EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.5, 0.5],
        id_key="chunk_id"
    )


# 对混合召回的候选重新评分，选出八条最终证据。
def create_evidence_retriever(chroma_store):
    # 先获取向量和关键词两路检索的候选集合。
    hybrid_retriever = create_hybrid_retriever(chroma_store)
    # 用交叉编码器比较问题与每条候选，保留得分最高的八条。
    rerank_model = HuggingFaceCrossEncoder(
        model_name="BAAI/bge-reranker-base",
    )
    reranker = CrossEncoderReranker(
        model=rerank_model,
        top_n=20,
    )

    # 把召回和重排接成一个可直接调用的检索器。
    evidence_retriever = ContextualCompressionRetriever(
        base_retriever=hybrid_retriever,
        base_compressor=reranker,
    )
    return evidence_retriever
