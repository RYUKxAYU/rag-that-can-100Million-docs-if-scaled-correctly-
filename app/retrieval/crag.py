from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

from app.retrieval.orchestrator import HybridRetrievalOrchestrator
from app.retrieval.schema import RetrievalItem, RetrievalResult


@dataclass(frozen=True)
class CRAGEvaluationResult:
    confidence_score: float
    classification: str
    reason: str
    top_score: float
    average_score: float
    hallucination_risk: float
    sufficient: bool


@dataclass(frozen=True)
class CorrectiveRetrievalOutcome:
    original_query: str
    final_query: str
    final_results: List[RetrievalResult]
    initial_assessment: CRAGEvaluationResult
    final_assessment: CRAGEvaluationResult
    corrected: bool
    rejected: bool


class RetrievalSufficiencyClassifier:
    def classify(self, confidence_score: float, top_score: float, hallucination_risk: float) -> str:
        if confidence_score >= 0.55 and top_score >= 0.08 and hallucination_risk <= 0.45:
            return "sufficient"
        if confidence_score >= 0.35 and top_score >= 0.05:
            return "partial"
        return "insufficient"


class RetrievalQualityEvaluator:
    def __init__(self, threshold: float = 0.55) -> None:
        self.threshold = threshold
        self.classifier = RetrievalSufficiencyClassifier()

    def evaluate(self, results: Sequence[RetrievalResult], query: str) -> CRAGEvaluationResult:
        if not results:
            return CRAGEvaluationResult(
                confidence_score=0.0,
                classification="insufficient",
                reason="no_results",
                top_score=0.0,
                average_score=0.0,
                hallucination_risk=1.0,
                sufficient=False,
            )

        top_score = max(result.score for result in results)
        average_score = sum(result.score for result in results) / len(results)
        query_terms = self._tokenize(query)
        coverage = self._compute_coverage(query_terms, results)
        hallucination_risk = self._estimate_hallucination_risk(results, coverage)
        confidence_score = self._compute_confidence(top_score, average_score, coverage)
        classification = self.classifier.classify(confidence_score, top_score, hallucination_risk)
        reason = self._format_reason(classification, confidence_score, top_score, hallucination_risk)

        return CRAGEvaluationResult(
            confidence_score=round(confidence_score, 3),
            classification=classification,
            reason=reason,
            top_score=round(top_score, 3),
            average_score=round(average_score, 3),
            hallucination_risk=round(hallucination_risk, 3),
            sufficient=classification == "sufficient",
        )

    def _tokenize(self, text: str) -> List[str]:
        return [token.lower() for token in re.findall(r"[a-zA-Z0-9]+", text) if token.strip()]

    def _compute_coverage(self, query_terms: List[str], results: Sequence[RetrievalResult]) -> float:
        if not query_terms:
            return 0.0
        matched_terms = {term for term in query_terms if any(term in result.item.text.lower() for result in results)}
        return len(matched_terms) / len(set(query_terms))

    def _estimate_hallucination_risk(self, results: Sequence[RetrievalResult], coverage: float) -> float:
        missing_source = sum(1 for result in results if not self._has_valid_source(result.item))
        low_score = sum(1 for result in results if result.score < 0.08)
        source_penalty = missing_source / len(results)
        score_penalty = low_score / len(results)
        risk = 1.0 - coverage
        risk += source_penalty * 0.35
        risk += score_penalty * 0.15
        return min(max(risk, 0.0), 1.0)

    def _compute_confidence(self, top_score: float, average_score: float, coverage: float) -> float:
        return min(1.0, top_score * 0.45 + average_score * 0.25 + coverage * 0.25 + 0.05)

    def _has_valid_source(self, item: RetrievalItem) -> bool:
        source = item.metadata.get("source") if isinstance(item.metadata, dict) else None
        if not source:
            return False
        return "unknown" not in str(source).lower()

    def _format_reason(self, classification: str, confidence_score: float, top_score: float, hallucination_risk: float) -> str:
        if classification == "sufficient":
            return "retrieval_sufficient"
        if classification == "partial":
            return "retrieval_partial"
        if hallucination_risk >= 0.5:
            return "hallucination_risk"
        if top_score < 0.05:
            return "low_top_score"
        return "insufficient"


class RetrievalRejectionSystem:
    def __init__(self, min_score: float = 0.05, hallucination_threshold: float = 0.5) -> None:
        self.min_score = min_score
        self.hallucination_threshold = hallucination_threshold

    def filter_results(self, results: Sequence[RetrievalResult], evaluation: CRAGEvaluationResult) -> List[RetrievalResult]:
        if evaluation.classification == "insufficient" and evaluation.hallucination_risk >= self.hallucination_threshold:
            return []

        filtered = []
        for result in results:
            if result.score < self.min_score:
                continue
            if not self._has_valid_source(result.item):
                continue
            filtered.append(result)

        return filtered

    def should_reject(self, evaluation: CRAGEvaluationResult) -> bool:
        return evaluation.classification == "insufficient" and evaluation.hallucination_risk >= self.hallucination_threshold

    def _has_valid_source(self, item: RetrievalItem) -> bool:
        source = item.metadata.get("source") if isinstance(item.metadata, dict) else None
        return bool(source) and "unknown" not in str(source).lower()


class CorrectiveRetrievalManager:
    def __init__(
        self,
        retriever: HybridRetrievalOrchestrator,
        evaluator: Optional[RetrievalQualityEvaluator] = None,
        rejector: Optional[RetrievalRejectionSystem] = None,
        max_attempts: int = 2,
    ) -> None:
        self.retriever = retriever
        self.evaluator = evaluator or RetrievalQualityEvaluator()
        self.rejector = rejector or RetrievalRejectionSystem()
        self.max_attempts = max_attempts

    async def run(
        self,
        query: str,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> CorrectiveRetrievalOutcome:
        original_query = query.strip()
        initial_results = await self.retriever.retrieve(
            query_embedding=query_embedding,
            query_text=original_query,
            top_k=top_k,
            use_bm25=True,
            use_vector=True,
        )
        initial_assessment = self.evaluator.evaluate(initial_results, original_query)
        filtered_results = self.rejector.filter_results(initial_results, initial_assessment)

        if initial_assessment.sufficient and filtered_results:
            return CorrectiveRetrievalOutcome(
                original_query=original_query,
                final_query=original_query,
                final_results=filtered_results,
                initial_assessment=initial_assessment,
                final_assessment=initial_assessment,
                corrected=False,
                rejected=False,
            )

        final_results = filtered_results
        final_assessment = initial_assessment
        current_query = original_query

        for attempt in range(2, self.max_attempts + 1):
            current_query = self._rewrite_query(current_query, initial_assessment, len(initial_results))
            corrective_params = self._correction_strategy(initial_assessment)
            alternative_results = await self.retriever.retrieve(
                query_embedding=query_embedding,
                query_text=current_query,
                top_k=top_k,
                use_bm25=corrective_params["use_bm25"],
                use_vector=corrective_params["use_vector"],
            )
            final_assessment = self.evaluator.evaluate(alternative_results, current_query)
            final_results = self.rejector.filter_results(alternative_results, final_assessment)
            if final_assessment.sufficient and final_results:
                break
            initial_assessment = final_assessment
            initial_results = alternative_results

        rejected = self.rejector.should_reject(final_assessment) or not bool(final_results)
        return CorrectiveRetrievalOutcome(
            original_query=original_query,
            final_query=current_query,
            final_results=final_results,
            initial_assessment=initial_assessment,
            final_assessment=final_assessment,
            corrected=current_query != original_query,
            rejected=rejected,
        )

    def _rewrite_query(self, query: str, evaluation: CRAGEvaluationResult, result_count: int) -> str:
        base = query.strip()
        if not base:
            return "Provide a clear, evidence-focused query for retrieval."

        if evaluation.classification == "insufficient":
            return f"{base} Include exact evidence, authoritative sources, and direct entity relationships."

        if evaluation.classification == "partial":
            return f"{base} Focus on the most relevant factual evidence and citation sources."

        return f"{base} Emphasize correctness and citation alignment."

    def _correction_strategy(self, evaluation: CRAGEvaluationResult) -> dict[str, bool]:
        if evaluation.classification == "partial":
            return {"use_bm25": True, "use_vector": False}
        if evaluation.classification == "insufficient":
            return {"use_bm25": False, "use_vector": True}
        return {"use_bm25": True, "use_vector": True}
