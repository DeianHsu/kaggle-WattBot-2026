# 为多事实题规划查询，组合向量和 BM25 关键词检索，再重排并汇总证据。

import re
from itertools import zip_longest

from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage


# 单条查询重排后和整题汇总后的证据块上限。
EVIDENCE_LIMIT = 20


# 把多事实题目拆成英文检索短语，便于分别查找各项证据。
SEARCH_PROMPT = """Plan short English queries to search research papers.
The original question will be searched separately.
For comparison, calculation, and cross-paper questions, write one query for
each independent fact needed to answer. Name the subject and metric in each.
Keep each fact's year, units, scope, and conditions. Retain the question's
terms and add a common English variant when useful.
Return only search phrases. Do not guess answers, values, or papers that
were not given. Return an empty list only if the question is already in
English and clearly asks for a single fact. If the question is not in
English, always return at least one English search phrase.
Return JSON in this format.
{"queries": ["short search phrase"]}"""


# 组合语义相似度和关键词匹配，兼顾同义表述与原文术语。
def create_hybrid_retriever(chroma_store):
    # 读取全部索引文本，供关键词检索器建立词频统计。
    chunk_ids = chroma_store.get(include=[])["ids"]
    if not chunk_ids:
        raise ValueError("索引为空，请先完成建库")
    chunks = chroma_store.get_by_ids(chunk_ids)
    vector_retriever = chroma_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 30},
    )
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


# 创建重排器，供服务热刷新混合检索索引时复用模型实例。
def create_reranker():
    rerank_model = HuggingFaceCrossEncoder(
        model_name="BAAI/bge-reranker-base",
    )
    return CrossEncoderReranker(
        model=rerank_model,
        top_n=EVIDENCE_LIMIT,
    )


# 创建检索器，在收到问题后执行召回和重排。
def create_evidence_retriever(chroma_store, reranker=None):
    hybrid_retriever = create_hybrid_retriever(chroma_store)
    if reranker is None:
        reranker = create_reranker()

    evidence_retriever = ContextualCompressionRetriever(
        base_retriever=hybrid_retriever,
        base_compressor=reranker,
    )
    return evidence_retriever


# 合并原问题和模型规划的子查询，检索并汇总本题的证据。
def retrieve_evidence(question, retriever, llm, answer_unit=""):
    messages = [
        SystemMessage(content=SEARCH_PROMPT),
        HumanMessage(content=f"{question}\nRequested unit: {answer_unit}"),
    ]
    queries = [question]
    # 规划解析失败时仍用原问题检索。
    try:
        plan = llm.invoke(messages)
    except OutputParserException:
        return retrieve_queries(retriever, queries)

    # 只接收格式有效的查询列表，并按首次出现的顺序去重。
    subqueries = plan.get("queries") if isinstance(plan, dict) else None
    if isinstance(subqueries, list) and all(
        isinstance(query, str) for query in subqueries
    ):
        queries.extend(query.strip() for query in subqueries if query.strip())
    queries = list(dict.fromkeys(queries))
    return retrieve_queries(retriever, queries)


# 将多条查询的检索结果合并为本题最终使用的证据。
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
