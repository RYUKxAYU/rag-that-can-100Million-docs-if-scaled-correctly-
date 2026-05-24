import asyncio
import tempfile
from pathlib import Path

from app.retrieval.bm25 import BM25Retriever
from app.retrieval.embeddings import BGEM3EmbeddingService
from app.retrieval.fusion import rrf_fusion
from app.retrieval.orchestrator import HybridRetrievalOrchestrator
from app.retrieval.schema import RetrievalItem
from app.retrieval.vector_store import VectorStore, VectorRetriever


def _sample_items():
    return [
        RetrievalItem(id="doc1", text="The quick brown fox jumps over the lazy dog.", metadata={"source": "doc1"}),
        RetrievalItem(id="doc2", text="A fast brown fox leaps across sleepy canines.", metadata={"source": "doc2"}),
        RetrievalItem(id="doc3", text="Explicit retrieval is deterministic and precise.", metadata={"source": "doc3"}),
    ]


def test_bge_m3_embedding_service_deterministic():
    service = BGEM3EmbeddingService(batch_size=2, embedding_dim=32)
    embedding_a = asyncio.run(service.embed_query("test query"))
    embedding_b = asyncio.run(service.embed_query("test query"))
    assert embedding_a == embedding_b
    assert len(embedding_a) == 32
    assert abs(sum(x * x for x in embedding_a) - 1.0) < 1e-6


def test_bm25_retriever_ranks_relevant_documents():
    items = _sample_items()
    retriever = BM25Retriever(items)
    results = retriever.retrieve("quick fox", top_k=2)
    assert len(results) == 2
    assert results[0].item.id == "doc1"
    assert results[0].score >= results[1].score


def test_vector_store_persistence_and_query():
    items = _sample_items()
    service = BGEM3EmbeddingService(batch_size=1, embedding_dim=16)
    embeddings = asyncio.run(service.embed_documents([item.text for item in items]))
    with tempfile.TemporaryDirectory() as temp_dir:
        store_path = Path(temp_dir) / "vectors.db"
        with VectorStore(str(store_path), embedding_dim=16) as store:
            store.upsert(items, embeddings)
        with VectorStore(str(store_path), embedding_dim=16) as reloaded:
            query_embedding = embeddings[0]
            results = reloaded.query(query_embedding, top_k=1)
            assert results[0].item.id == "doc1"
            assert results[0].score > 0


def test_bm25_retriever_empty_query_returns_no_results():
    items = _sample_items()
    retriever = BM25Retriever(items)
    assert retriever.retrieve("", top_k=5) == []


def test_rrf_fusion_combines_rankings_consistently():
    items = _sample_items()
    bm25_results = [
        type("R", (), {"item": items[0], "score": 2.0, "rank": 1, "source": "bm25"}),
        type("R", (), {"item": items[2], "score": 1.0, "rank": 2, "source": "bm25"}),
    ]
    vector_results = [
        type("R", (), {"item": items[1], "score": 2.0, "rank": 1, "source": "vector"}),
        type("R", (), {"item": items[0], "score": 1.0, "rank": 2, "source": "vector"}),
    ]
    fused = rrf_fusion(bm25_results, vector_results, top_k=3, k=50)
    assert len(fused) == 3
    assert fused[0].item.id == "doc1"
    assert fused[0].source == "hybrid"
    assert fused[0].score >= fused[1].score


def test_hybrid_retrieval_orchestrator_async():
    items = _sample_items()
    bm25 = BM25Retriever(items)
    service = BGEM3EmbeddingService(batch_size=1, embedding_dim=16)
    embeddings = asyncio.run(service.embed_documents([item.text for item in items]))
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
                )
            )
            assert len(results) >= 1
            assert results[0].source == "hybrid"
            assert results[0].rank == 1


def test_orchestrator_vector_only():
    items = _sample_items()
    service = BGEM3EmbeddingService(batch_size=1, embedding_dim=16)
    embeddings = asyncio.run(service.embed_documents([item.text for item in items]))
    with tempfile.TemporaryDirectory() as temp_dir:
        with VectorStore(str(Path(temp_dir) / "vector.db"), embedding_dim=16) as store:
            store.upsert(items, embeddings)
            vector_retriever = VectorRetriever(store)
            orchestrator = HybridRetrievalOrchestrator(BM25Retriever(items), vector_retriever)
            query_embedding = asyncio.run(service.embed_query("retrieval test"))
            results = asyncio.run(
                orchestrator.retrieve(
                    query_embedding=query_embedding,
                    query_text=None,
                    top_k=2,
                    use_bm25=False,
                    use_vector=True,
                )
            )
            assert results[0].source == "vector"
            assert results[0].rank == 1


def test_orchestrator_bm25_only():
    items = _sample_items()
    bm25 = BM25Retriever(items)
    service = BGEM3EmbeddingService(batch_size=1, embedding_dim=16)
    query_embedding = asyncio.run(service.embed_query("quick brown"))
    with tempfile.TemporaryDirectory() as temp_dir:
        with VectorStore(str(Path(temp_dir) / "vector.db"), embedding_dim=16) as store:
            vector_retriever = VectorRetriever(store)
            orchestrator = HybridRetrievalOrchestrator(bm25, vector_retriever)
            results = asyncio.run(
                orchestrator.retrieve(
                    query_embedding=query_embedding,
                    query_text="quick brown",
                    top_k=2,
                    use_bm25=True,
                    use_vector=False,
                )
            )
            assert results[0].source == "bm25"
            assert results[0].rank == 1


def test_vector_store_embedding_count_mismatch_raises():
    items = _sample_items()
    service = BGEM3EmbeddingService(batch_size=1, embedding_dim=16)
    embeddings = asyncio.run(service.embed_documents([item.text for item in items[:2]]))
    with tempfile.TemporaryDirectory() as temp_dir:
        with VectorStore(str(Path(temp_dir) / "vector.db"), embedding_dim=16) as store:
            try:
                store.upsert(items, embeddings)
            except ValueError as exc:
                assert "number of embeddings must match" in str(exc).lower()
            else:
                assert False, "Expected ValueError for mismatched embedding count"


def test_vector_store_iterate_vectors_returns_expected_items():
    items = _sample_items()
    service = BGEM3EmbeddingService(batch_size=1, embedding_dim=16)
    embeddings = asyncio.run(service.embed_documents([item.text for item in items]))
    with tempfile.TemporaryDirectory() as temp_dir:
        with VectorStore(str(Path(temp_dir) / "vector.db"), embedding_dim=16) as store:
            store.upsert(items, embeddings)
            stored_items = list(store._iterate_vectors())
            assert {item.id for item in stored_items} == {"doc1", "doc2", "doc3"}


def test_vector_store_zero_query_embedding_returns_empty():
    items = _sample_items()
    service = BGEM3EmbeddingService(batch_size=1, embedding_dim=16)
    embeddings = asyncio.run(service.embed_documents([item.text for item in items]))
    with tempfile.TemporaryDirectory() as temp_dir:
        with VectorStore(str(Path(temp_dir) / "vector.db"), embedding_dim=16) as store:
            store.upsert(items, embeddings)
            results = store.query([0.0] * 16, top_k=3)
            assert results == []


def test_vector_store_invalid_embedding_length_raises():
    items = _sample_items()
    service = BGEM3EmbeddingService(batch_size=1, embedding_dim=16)
    embeddings = asyncio.run(service.embed_documents([item.text for item in items]))
    with tempfile.TemporaryDirectory() as temp_dir:
        with VectorStore(str(Path(temp_dir) / "vector.db"), embedding_dim=8) as store:
            try:
                store.upsert(items, embeddings)
            except ValueError as exc:
                assert "Embedding length does not match" in str(exc)
            else:
                assert False, "Expected ValueError for invalid embedding length"


def test_vector_store_query_length_mismatch_raises():
    items = _sample_items()
    service = BGEM3EmbeddingService(batch_size=1, embedding_dim=16)
    embeddings = asyncio.run(service.embed_documents([item.text for item in items]))
    with tempfile.TemporaryDirectory() as temp_dir:
        with VectorStore(str(Path(temp_dir) / "vector.db"), embedding_dim=16) as store:
            store.upsert(items, embeddings)
            try:
                store.query([0.0] * 8, top_k=1)
            except ValueError as exc:
                assert "Query embedding length does not match" in str(exc)
            else:
                assert False, "Expected ValueError for query embedding length mismatch"


def test_embedding_service_batching():
    service = BGEM3EmbeddingService(batch_size=2, embedding_dim=16)
    embeddings = asyncio.run(service.embed_documents(["a", "b", "c"]))
    assert len(embeddings) == 3
    assert all(len(vector) == 16 for vector in embeddings)


def test_orchestrator_no_methods_returns_empty():
    items = _sample_items()
    bm25 = BM25Retriever(items)
    service = BGEM3EmbeddingService(batch_size=1, embedding_dim=16)
    query_embedding = asyncio.run(service.embed_query("test"))
    with tempfile.TemporaryDirectory() as temp_dir:
        with VectorStore(str(Path(temp_dir) / "vector.db"), embedding_dim=16) as store:
            orchestrator = HybridRetrievalOrchestrator(bm25, VectorRetriever(store))
            results = asyncio.run(
                orchestrator.retrieve(
                    query_embedding=query_embedding,
                    query_text=None,
                    top_k=2,
                    use_bm25=False,
                    use_vector=False,
                )
            )
            assert results == []
