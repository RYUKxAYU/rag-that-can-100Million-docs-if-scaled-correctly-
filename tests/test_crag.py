import asyncio
import tempfile
from pathlib import Path
from typing import List

from app.retrieval import (
    BGEM3EmbeddingService,
    BM25Retriever,
    CorrectiveRetrievalManager,
    HybridRetrievalOrchestrator,
    RetrievalQualityEvaluator,
    RetrievalRejectionSystem,
    VectorRetriever,
    VectorStore,
)
from app.retrieval.crag import CRAGEvaluationResult, CorrectiveRetrievalOutcome
from app.retrieval.schema import RetrievalItem, RetrievalResult


def _sample_items() -> List[RetrievalItem]:
    return [
        RetrievalItem(id="doc1", text="The supply chain impact on pricing is significant.", metadata={"source": "doc1"}),
        RetrievalItem(id="doc2", text="Financial models show reported inflation trends.", metadata={"source": "doc2"}),
        RetrievalItem(id="doc3", text="Unrelated content about gardening and plants.", metadata={"source": "doc3"}),
    ]


def test_retrieval_quality_evaluator_classifies_sufficient_results() -> None:
    items = _sample_items()
    results = [RetrievalResult(item=items[0], score=0.85, rank=1, source="hybrid")]
    evaluator = RetrievalQualityEvaluator()
    assessment = evaluator.evaluate(results, "supply chain pricing")

    assert isinstance(assessment, CRAGEvaluationResult)
    assert assessment.classification == "sufficient"
    assert assessment.sufficient is True
    assert assessment.hallucination_risk < 0.5


def test_retrieval_quality_evaluator_detects_partial_quality() -> None:
    items = _sample_items()
    results = [RetrievalResult(item=items[2], score=0.12, rank=1, source="bm25")]
    evaluator = RetrievalQualityEvaluator()
    assessment = evaluator.evaluate(results, "supply chain pricing")

    assert assessment.classification in {"partial", "insufficient"}
    assert assessment.confidence_score <= 0.5


def test_retrieval_rejection_system_filters_hallucination_prone_results() -> None:
    items = _sample_items()
    results = [
        RetrievalResult(item=items[0], score=0.02, rank=1, source="vector"),
        RetrievalResult(item=RetrievalItem(id="doc4", text="Fake content.", metadata={}), score=0.5, rank=2, source="bm25"),
    ]
    evaluator = RetrievalQualityEvaluator()
    assessment = evaluator.evaluate(results, "supply chain pricing")
    rejector = RetrievalRejectionSystem(min_score=0.05)

    filtered = rejector.filter_results(results, assessment)
    assert all(item.score >= 0.05 for item in filtered)
    assert all(result.item.metadata.get("source") for result in filtered)


def test_corrective_retrieval_manager_uses_alternative_strategy() -> None:
    items = _sample_items()
    bm25 = BM25Retriever(items)
    service = BGEM3EmbeddingService(batch_size=1, embedding_dim=16)
    embeddings = asyncio.run(service.embed_documents([item.text for item in items]))

    with tempfile.TemporaryDirectory() as temp_dir:
        with VectorStore(str(Path(temp_dir) / "vector.db"), embedding_dim=16) as store:
            store.upsert(items, embeddings)
            vector_retriever = VectorRetriever(store)
            manager = CorrectiveRetrievalManager(
                retriever=HybridRetrievalOrchestrator(bm25, vector_retriever),
                evaluator=RetrievalQualityEvaluator(threshold=0.95),
                rejector=RetrievalRejectionSystem(min_score=0.05),
                max_attempts=2,
            )
            outcome = asyncio.run(
                manager.run(
                    query="Describe the supply chain pricing relationship",
                    query_embedding=embeddings[0],
                    top_k=2,
                )
            )

    assert isinstance(outcome, CorrectiveRetrievalOutcome)
    assert outcome.corrected is True
    assert isinstance(outcome.final_assessment, CRAGEvaluationResult)
    assert outcome.rejected is False


def test_corrective_retrieval_manager_rejects_low_confidence_chains() -> None:
    items = [
        RetrievalItem(id="doc1", text="Unknown lorem ipsum.", metadata={}),
        RetrievalItem(id="doc2", text="Unknown dolor sit amet.", metadata={}),
    ]
    bm25 = BM25Retriever(items)
    service = BGEM3EmbeddingService(batch_size=1, embedding_dim=16)
    embeddings = asyncio.run(service.embed_documents([item.text for item in items]))

    with tempfile.TemporaryDirectory() as temp_dir:
        with VectorStore(str(Path(temp_dir) / "vector.db"), embedding_dim=16) as store:
            store.upsert(items, embeddings)
            vector_retriever = VectorRetriever(store)
            manager = CorrectiveRetrievalManager(
                retriever=HybridRetrievalOrchestrator(bm25, vector_retriever),
                evaluator=RetrievalQualityEvaluator(threshold=0.80),
                rejector=RetrievalRejectionSystem(min_score=0.05, hallucination_threshold=0.4),
                max_attempts=2,
            )
            outcome = asyncio.run(
                manager.run(
                    query="Given facts",
                    query_embedding=embeddings[0],
                    top_k=2,
                )
            )

    assert outcome.rejected is True
    assert outcome.final_results == []
