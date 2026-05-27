from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from app.retrieval.bm25 import BM25Retriever
from app.retrieval.embeddings import BGEM3EmbeddingService, EmbeddingService
from app.retrieval.orchestrator import HybridRetrievalOrchestrator
from app.retrieval.schema import RetrievalResult


@dataclass(frozen=True)
class HypotheticalDocument:
    query: str
    text: str
    metadata: Dict[str, str]


@dataclass(frozen=True)
class SemanticExpansionValidation:
    valid: bool
    issues: List[str]


@dataclass(frozen=True)
class HyDEExpansionOutcome:
    query: str
    hyde_documents: List[HypotheticalDocument]
    original_embedding: List[float]
    fused_embedding: List[float]
    retrieval_results: List[RetrievalResult]
    validation: SemanticExpansionValidation


class HypotheticalDocumentGenerator:
    def __init__(self, max_documents: int = 3) -> None:
        self.max_documents = max_documents
        self.templates = [
            "{query} is supported by deterministic evidence about {topic} and factual context.",
            "A semantic expansion of {query} includes related details on {topic} and evidence-based recall.",
            "Build on {query} with concrete examples of {topic} and retrieval precision.",
        ]

    def generate(self, query: str) -> List[HypotheticalDocument]:
        query = str(query).strip()
        tokens = self._tokenize(query)
        topic = self._derive_topic(tokens)
        seed = hashlib.blake2b(query.encode("utf-8"), digest_size=8).hexdigest()

        documents: List[HypotheticalDocument] = []
        for index in range(self.max_documents):
            template = self.templates[index % len(self.templates)]
            text = template.format(query=query, topic=topic)
            metadata = {"source": f"hyde_generated_{seed}_{index}", "hyde_index": str(index)}
            documents.append(HypotheticalDocument(query=query, text=text, metadata=metadata))

        return documents

    def _tokenize(self, text: str) -> List[str]:
        return [token.lower() for token in re.findall(r"[a-zA-Z0-9]+", text) if token]

    def _derive_topic(self, tokens: List[str]) -> str:
        if not tokens:
            return "context"
        return " ".join(tokens[: min(3, len(tokens))])


class HyDEEmbeddingService(EmbeddingService):
    def __init__(
        self,
        embedding_dim: int = 1536,
        batch_size: int = 16,
        model_path: Optional[str] = None,
    ):
        self.base_service = BGEM3EmbeddingService(embedding_dim=embedding_dim, batch_size=batch_size, model_path=model_path)

    async def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return await self.base_service.embed_documents(texts)

    async def embed_query(self, query: str) -> List[float]:
        return await self.base_service.embed_query(query)

    async def embed_hypothetical_documents(self, documents: Sequence[HypotheticalDocument]) -> List[List[float]]:
        texts = [self._prepare_text(document) for document in documents]
        return await self.base_service.embed_documents(texts)

    def _prepare_text(self, document: HypotheticalDocument) -> str:
        return f"{document.text} [HyDE] {document.query}"


class EmbeddingFusionEngine:
    def fuse_embeddings(
        self,
        query_embedding: List[float],
        hyde_embeddings: Sequence[List[float]],
        query_weight: float = 0.6,
        hyde_weight: float = 0.4,
    ) -> List[float]:
        if not query_embedding:
            return []
        if not hyde_embeddings:
            return list(query_embedding)

        combined_hyde = [0.0] * len(query_embedding)
        for hyde_embedding in hyde_embeddings:
            for index, value in enumerate(hyde_embedding):
                combined_hyde[index] += value
        for index in range(len(combined_hyde)):
            combined_hyde[index] /= len(hyde_embeddings)

        fused = [
            query_embedding[index] * query_weight + combined_hyde[index] * hyde_weight
            for index in range(len(query_embedding))
        ]
        return self._normalize(fused)

    def _normalize(self, vector: List[float]) -> List[float]:
        length = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / length for value in vector]


class SemanticExpansionValidator:
    def validate(
        self,
        query: str,
        hyde_documents: Sequence[HypotheticalDocument],
        fused_embedding: Sequence[float],
    ) -> SemanticExpansionValidation:
        issues: List[str] = []
        query_terms = set(self._tokenize(query))

        if not query_terms:
            issues.append("empty_query")
        if not hyde_documents:
            issues.append("no_hypothetical_documents")

        for document in hyde_documents:
            document_terms = set(self._tokenize(document.text))
            if not query_terms.intersection(document_terms):
                issues.append("document_missing_query_terms")
                break

        if not fused_embedding:
            issues.append("empty_fused_embedding")
        elif abs(math.sqrt(sum(value * value for value in fused_embedding)) - 1.0) > 1e-6:
            issues.append("unnormalized_embedding")

        return SemanticExpansionValidation(valid=len(issues) == 0, issues=issues)

    def _tokenize(self, text: str) -> List[str]:
        return [token.lower() for token in re.findall(r"[a-zA-Z0-9]+", text) if token]


class HyDEQueryExpander:
    def __init__(
        self,
        generator: Optional[HypotheticalDocumentGenerator] = None,
        hyde_service: Optional[HyDEEmbeddingService] = None,
        fusion_engine: Optional[EmbeddingFusionEngine] = None,
        validator: Optional[SemanticExpansionValidator] = None,
    ) -> None:
        self.generator = generator or HypotheticalDocumentGenerator()
        self.hyde_service = hyde_service or HyDEEmbeddingService()
        self.fusion_engine = fusion_engine or EmbeddingFusionEngine()
        self.validator = validator or SemanticExpansionValidator()

    async def expand(
        self,
        query: str,
        retriever: HybridRetrievalOrchestrator,
        top_k: int = 5,
        use_bm25: bool = True,
        use_vector: bool = True,
    ) -> HyDEExpansionOutcome:
        self._align_embedding_dimension_with_retriever(retriever)
        hyde_documents = self.generator.generate(query)
        original_embedding = await self.hyde_service.embed_query(query)
        hyde_embeddings = await self.hyde_service.embed_hypothetical_documents(hyde_documents)
        fused_embedding = self.fusion_engine.fuse_embeddings(original_embedding, hyde_embeddings)
        retrieval_results = await retriever.retrieve(
            query_embedding=fused_embedding,
            query_text=query,
            top_k=top_k,
            bm25_k=top_k,
            vector_k=top_k,
            use_bm25=use_bm25,
            use_vector=use_vector,
        )
        validation = self.validator.validate(query, hyde_documents, fused_embedding)

        return HyDEExpansionOutcome(
            query=query,
            hyde_documents=hyde_documents,
            original_embedding=original_embedding,
            fused_embedding=fused_embedding,
            retrieval_results=retrieval_results,
            validation=validation,
        )

    def _align_embedding_dimension_with_retriever(self, retriever: HybridRetrievalOrchestrator) -> None:
        vector_retriever = getattr(retriever, "vector_retriever", None)
        store = getattr(vector_retriever, "store", None)
        embedding_dim = getattr(store, "embedding_dim", None)
        if embedding_dim and self.hyde_service.base_service.embedding_dim != embedding_dim:
            self.hyde_service = HyDEEmbeddingService(
                embedding_dim=embedding_dim,
                batch_size=self.hyde_service.base_service.batch_size,
                model_path=self.hyde_service.base_service.model_path,
            )
