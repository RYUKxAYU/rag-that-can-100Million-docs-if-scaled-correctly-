import asyncio
from pathlib import Path
import tempfile

import pytest

from app.retrieval.bm25 import BM25Retriever
from app.retrieval.embeddings import BGEM3EmbeddingService
from app.retrieval.orchestrator import HybridRetrievalOrchestrator
from app.retrieval.reranker import (
    CrossEncoderReranker,
    ContextSelector,
    count_tokens,
    normalize_scores,
)
from app.retrieval.schema import RetrievalItem, RetrievalResult
from app.retrieval.vector_store import VectorStore, VectorRetriever


def _sample_items():
    return [
        RetrievalItem(id="doc1", text="The quick brown fox jumps over the lazy dog.", metadata={"source": "doc1"}),
        RetrievalItem(id="doc2", text="A fast brown fox leaps across sleepy canines.", metadata={"source": "doc2"}),
        RetrievalItem(id="doc3", text="Explicit retrieval is deterministic and precise.", metadata={"source": "doc3"}),
    ]


def test_count_tokens_treats_text_as_words():
    assert count_tokens("Hello, world! This is 4 tokens.") == 6


def test_normalize_scores_scales_values_to_unit_range():
    normalized = normalize_scores([0.2, 0.5, 0.7])
    assert normalized[0] == 0.0
    assert normalized[1] == pytest.approx(0.6)
    assert normalized[2] == 1.0


def test_context_selector_obeys_token_budget():
    items = _sample_items()
    results = [RetrievalResult(item=item, score=1.0, rank=index + 1, source="bm25") for index, item in enumerate(items)]
    selector = ContextSelector(token_budget=12)
    selected = selector.select("quick fox", results)
    assert len(selected) == 1
    assert selected[0].item.id == "doc1"


def test_cross_encoder_reranker_filters_below_threshold():
    items = _sample_items()
    results = [RetrievalResult(item=item, score=1.0, rank=index + 1, source="bm25") for index, item in enumerate(items)]
    reranker = CrossEncoderReranker(token_budget=50, threshold=0.35)
    reranked = reranker.rerank("brown fox", results, top_k=3)
    assert all(result.score >= 0.35 for result in reranked)
    assert all(result.source == "reranker" for result in reranked)


def test_cross_encoder_reranker_integration_with_vector_store():
    items = _sample_items()
    bm25 = BM25Retriever(items)
    service = BGEM3EmbeddingService(batch_size=1, embedding_dim=16)
    embeddings = asyncio.run(service.embed_documents([item.text for item in items]))
    reranker = CrossEncoderReranker(token_budget=50, threshold=0.0)

    with tempfile.TemporaryDirectory() as temp_dir:
        with VectorStore(str(Path(temp_dir) / "vector.db"), embedding_dim=16) as store:
            store.upsert(items, embeddings)
            vector_results = asyncio.run(VectorRetriever(store).retrieve(embeddings[0], top_k=3))
            bm25_results = bm25.retrieve("quick brown", top_k=3)
            combined = bm25_results + vector_results
            reranked = reranker.rerank("quick brown", combined, top_k=3)
            assert len(reranked) > 0
            assert reranked[0].source == "reranker"


def test_hybrid_orchestrator_reranks_with_cross_encoder():
    items = _sample_items()
    bm25 = BM25Retriever(items)
    service = BGEM3EmbeddingService(batch_size=1, embedding_dim=16)
    embeddings = asyncio.run(service.embed_documents([item.text for item in items]))
    reranker = CrossEncoderReranker(token_budget=100, threshold=0.0)

    with tempfile.TemporaryDirectory() as temp_dir:
        with VectorStore(str(Path(temp_dir) / "vector.db"), embedding_dim=16) as store:
            store.upsert(items, embeddings)
            vector_retriever = VectorRetriever(store)
            orchestrator = HybridRetrievalOrchestrator(bm25, vector_retriever)
            query_embedding = asyncio.run(service.embed_query("brown fox"))
            results = asyncio.run(
                orchestrator.retrieve(
                    query_embedding=query_embedding,
                    query_text="brown fox",
                    top_k=3,
                    bm25_k=3,
                    vector_k=3,
                    reranker=reranker,
                    rerank=True,
                    rerank_threshold=0.0,
                )
            )
            assert results[0].source == "reranker"
            assert len(results) > 0
