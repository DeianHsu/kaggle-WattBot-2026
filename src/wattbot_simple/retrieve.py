# 组合向量检索和 BM25 关键词检索，再用交叉编码器重排候选证据。

from itertools import zip_longest

from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever


EVIDENCE_LIMIT = 20


def create_hybrid_retriever(chroma_store, bm25_retriever):
    vector_retriever = chroma_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 30},
    )
    # 合并两路候选排名，并按文本块标识去重。
    return EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.5, 0.5],
        id_key="chunk_id"
    )


# 创建检索器，在收到问题后执行召回和重排。
def create_evidence_retriever(chroma_store, bm25_retriever):
    hybrid_retriever = create_hybrid_retriever(chroma_store, bm25_retriever)
    rerank_model = HuggingFaceCrossEncoder(
        model_name="BAAI/bge-reranker-base",
    )
    reranker = CrossEncoderReranker(
        model=rerank_model,
        top_n=EVIDENCE_LIMIT,
    )

    evidence_retriever = ContextualCompressionRetriever(
        base_retriever=hybrid_retriever,
        base_compressor=reranker,
    )
    return evidence_retriever


def retrieve_queries(retriever, queries):
    results = [retriever.invoke(query) for query in queries]
    # 按查询轮流取证据，给排名靠前的不同事实各留位置。
    evidence = {}
    for group in zip_longest(*results):
        for chunk in group:
            if chunk is not None:
                evidence.setdefault(chunk.metadata["chunk_id"], chunk)
                if len(evidence) == EVIDENCE_LIMIT:
                    return list(evidence.values())
    return list(evidence.values())
