import asyncio
from typing import List, Optional

from app.retrieval.bm25 import BM25Retriever
from app.retrieval.fusion import rrf_fusion
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.schema import RetrievalResult
from app.retrieval.vector_store import VectorRetriever


class HybridRetrievalOrchestrator:
    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        vector_retriever: VectorRetriever,
        use_hybrid: bool = True,
    ):
        self.bm25_retriever = bm25_retriever
        self.vector_retriever = vector_retriever
        self.use_hybrid = use_hybrid

    async def retrieve(
        self,
        query_embedding: List[float],
        query_text: Optional[str] = None,
        top_k: int = 10,
        bm25_k: int = 10,
        vector_k: int = 10,
        use_bm25: bool = True,
        use_vector: bool = True,
        reranker: Optional[CrossEncoderReranker] = None,
        rerank: bool = False,
        rerank_threshold: float = 0.35,
        rerank_token_budget: int = 512,
    ) -> List[RetrievalResult]:
        if not use_bm25 and not use_vector:
            return []

        tasks = []
        if use_bm25 and query_text is not None:
            tasks.append(asyncio.to_thread(self.bm25_retriever.retrieve, query_text, bm25_k))
        if use_vector:
            tasks.append(self.vector_retriever.retrieve(query_embedding, top_k=vector_k))

        results = await asyncio.gather(*tasks)

        bm25_results = []
        vector_results = []
        if use_bm25 and query_text is not None:
            bm25_results = results[0] if use_vector else results[0]
            if use_vector:
                vector_results = results[1]
        elif use_vector:
            vector_results = results[0]

        if self.use_hybrid and use_bm25 and use_vector:
            results = rrf_fusion(bm25_results, vector_results, top_k=top_k)
        elif use_bm25 and not use_vector:
            results = bm25_results[:top_k]
        else:
            results = vector_results[:top_k]

        if rerank and reranker is not None and query_text is not None:
            return reranker.rerank(
                query_text,
                results,
                top_k=top_k,
                threshold=rerank_threshold,
                token_budget=rerank_token_budget,
            )

        return results
