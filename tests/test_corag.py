import asyncio
import tempfile
from pathlib import Path
from typing import List

from app.retrieval import (
    BM25Retriever,
    BGEM3EmbeddingService,
    HybridRetrievalOrchestrator,
    IterativeRetrievalLoop,
    QueryRewriter,
    RecursiveRetrievalOrchestrator,
    RetrievalCritic,
    RetrievalCriticResult,
    VectorRetriever,
    VectorStore,
)
from app.retrieval.schema import RetrievalItem, RetrievalResult


def _sample_items() -> List[RetrievalItem]:
    return [
        RetrievalItem(id="doc1", text="A quick brown fox jumps over a lazy dog.", metadata={"source": "doc1"}),
        RetrievalItem(id="doc2", text="Supply chain disruption affects pricing and delivery.", metadata={"source": "doc2"}),
        RetrievalItem(id="doc3", text="Iterative retrieval improves coverage for complex queries.", metadata={"source": "doc3"}),
    ]


def test_retrieval_critic_returns_low_confidence_for_empty_results() -> None:
    critic = RetrievalCritic()
    critique = critic.evaluate([], "What caused the event?")

    assert isinstance(critique, RetrievalCriticResult)
    assert critique.confidence_score == 0.0
    assert critique.is_sufficient is False
    assert critique.reason == "no_results"


def test_retrieval_critic_returns_high_confidence_for_strong_results() -> None:
    items = _sample_items()
    results = [RetrievalResult(item=items[0], score=0.9, rank=1, source="vector")]
    critic = RetrievalCritic(threshold=0.25)
    critique = critic.evaluate(results, "quick fox")

    assert critique.is_sufficient is True
    assert critique.confidence_score > 0.25
    assert critique.reason == "sufficient"


def test_query_rewriter_refines_query_after_low_confidence() -> None:
    critic_result = RetrievalCriticResult(
        confidence_score=0.2,
        is_sufficient=False,
        reason="low_confidence",
        top_score=0.05,
        average_score=0.02,
    )
    rewriter = QueryRewriter()
    refined = rewriter.rewrite("Explain the relationship between inflation and pricing", critic_result)

    assert "Provide the causal chain" in refined


def test_iterative_retrieval_loop_stops_at_max_iterations() -> None:
    items = _sample_items()
    bm25 = BM25Retriever(items)
    service = BGEM3EmbeddingService(batch_size=1, embedding_dim=16)
    embeddings = asyncio.run(service.embed_documents([item.text for item in items]))

    with tempfile.TemporaryDirectory() as temp_dir:
        with VectorStore(str(Path(temp_dir) / "vector.db"), embedding_dim=16) as store:
            store.upsert(items, embeddings)
            vector_retriever = VectorRetriever(store)
            orchestrator = IterativeRetrievalLoop(
                retriever=HybridRetrievalOrchestrator(bm25, vector_retriever),
                critic=RetrievalCritic(threshold=0.99),
                max_iterations=3,
            )
            result = asyncio.run(
                orchestrator.run(
                    query="Explain the connection between supply chain and pricing",
                    query_embedding=embeddings[0],
                    query_text="Explain the connection between supply chain and pricing",
                    top_k=2,
                )
            )

    assert result["iterations"] == 3
    assert result["sufficient"] is False
    assert result["final_query"] != ""


def test_recursive_retrieval_orchestrator_runs_multi_hop_cycle() -> None:
    items = _sample_items()
    bm25 = BM25Retriever(items)
    service = BGEM3EmbeddingService(batch_size=1, embedding_dim=16)
    embeddings = asyncio.run(service.embed_documents([item.text for item in items]))

    with tempfile.TemporaryDirectory() as temp_dir:
        with VectorStore(str(Path(temp_dir) / "vector.db"), embedding_dim=16) as store:
            store.upsert(items, embeddings)
            vector_retriever = VectorRetriever(store)
            orchestrator = RecursiveRetrievalOrchestrator(
                iterative_loop=IterativeRetrievalLoop(
                    retriever=HybridRetrievalOrchestrator(bm25, vector_retriever),
                    critic=RetrievalCritic(threshold=0.99),
                    max_iterations=2,
                ),
                max_depth=1,
            )
            summary = asyncio.run(
                orchestrator.recursive_retrieve(
                    query="Describe the relationship between supply chain and pricing",
                    query_embedding=embeddings[0],
                    classification="multi-hop",
                    query_text="Describe the relationship between supply chain and pricing",
                    top_k=2,
                )
            )

    assert summary["root"]["iterations"] >= 1
    assert summary["recursive_steps"][0]["depth"] == 0
    assert summary["child"] is not None
    assert isinstance(summary["child"]["final_results"], list)
