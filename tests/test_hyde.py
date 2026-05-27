import asyncio
import tempfile
from pathlib import Path
from typing import List

from app.retrieval import (
    BGEM3EmbeddingService,
    BM25Retriever,
    EmbeddingFusionEngine,
    HypotheticalDocumentGenerator,
    HyDEEmbeddingService,
    HyDEQueryExpander,
    HybridRetrievalOrchestrator,
    SemanticExpansionValidator,
    VectorRetriever,
    VectorStore,
)
from app.retrieval.schema import RetrievalItem


def _sample_items() -> List[RetrievalItem]:
    return [
        RetrievalItem(id="doc1", text="Semantic search and query expansion improve recall.", metadata={"source": "doc1"}),
        RetrievalItem(id="doc2", text="HyDE generates deterministic hypothetical evidence for better retrieval.", metadata={"source": "doc2"}),
        RetrievalItem(id="doc3", text="This unrelated document contains gardening information.", metadata={"source": "doc3"}),
    ]


def test_hypothetical_document_generation_is_deterministic() -> None:
    generator = HypotheticalDocumentGenerator(max_documents=3)
    docs_one = generator.generate("semantic search recall")
    docs_two = generator.generate("semantic search recall")

    assert [doc.text for doc in docs_one] == [doc.text for doc in docs_two]
    assert len(docs_one) == 3
    assert all(doc.query == "semantic search recall" for doc in docs_one)


def test_hyde_embedding_service_generates_normalized_embeddings() -> None:
    service = HyDEEmbeddingService(batch_size=2, embedding_dim=32)
    docs = HypotheticalDocumentGenerator(max_documents=2).generate("semantic search recall")
    embeddings = asyncio.run(service.embed_hypothetical_documents(docs))

    assert len(embeddings) == 2
    assert all(len(embedding) == 32 for embedding in embeddings)
    assert all(abs(sum(value * value for value in embedding) - 1.0) < 1e-6 for embedding in embeddings)


def test_embedding_fusion_combines_query_and_hyde_embeddings() -> None:
    service = HyDEEmbeddingService(batch_size=1, embedding_dim=16)
    original = asyncio.run(service.embed_query("semantic search recall"))
    docs = HypotheticalDocumentGenerator(max_documents=2).generate("semantic search recall")
    hyde_embeddings = asyncio.run(service.embed_hypothetical_documents(docs))

    fusion_engine = EmbeddingFusionEngine()
    fused = fusion_engine.fuse_embeddings(original, hyde_embeddings, query_weight=0.7, hyde_weight=0.3)

    assert len(fused) == len(original)
    assert abs(sum(value * value for value in fused) - 1.0) < 1e-6
    assert fused != original


def test_semantic_expansion_pipeline_returns_valid_retrieval_results() -> None:
    items = _sample_items()
    bm25 = BM25Retriever(items)
    service = HyDEEmbeddingService(batch_size=1, embedding_dim=16)
    embeddings = asyncio.run(service.embed_documents([item.text for item in items]))

    with tempfile.TemporaryDirectory() as temp_dir:
        with VectorStore(str(Path(temp_dir) / "vector.db"), embedding_dim=16) as store:
            store.upsert(items, embeddings)
            vector_retriever = VectorRetriever(store)
            retriever = HybridRetrievalOrchestrator(bm25, vector_retriever)
            expander = HyDEQueryExpander()
            outcome = asyncio.run(expander.expand("semantic search recall", retriever, top_k=2))

    assert outcome.validation.valid is True
    assert outcome.hyde_documents
    assert outcome.retrieval_results
    assert len(outcome.fused_embedding) == len(outcome.original_embedding)


def test_semantic_expansion_validator_detects_invalid_hypotheticals() -> None:
    validator = SemanticExpansionValidator()
    from app.retrieval.hyde import HypotheticalDocument

    docs = [HypotheticalDocument(query="", text="unrelated content", metadata={"source": "unknown"})]
    validation = validator.validate("", docs, [])

    assert validation.valid is False
    assert "empty_query" in validation.issues
    assert "no_hypothetical_documents" not in validation.issues
