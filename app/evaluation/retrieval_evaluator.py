from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from app.retrieval.schema import RetrievalResult


@dataclass(frozen=True)
class RetrievalEvaluationMetrics:
    precision_at_k: float
    mean_reciprocal_rank: float
    citation_accuracy: float


def precision_at_k(results: Sequence[RetrievalResult], relevant_ids: Iterable[str], k: int = 10) -> float:
    if k <= 0:
        return 0.0
    relevant_set = set(relevant_ids)
    top_results = results[:k]
    if not top_results:
        return 0.0
    hits = sum(1 for result in top_results if result.item.id in relevant_set)
    return hits / len(top_results)


def mean_reciprocal_rank(results: Sequence[RetrievalResult], relevant_ids: Iterable[str]) -> float:
    relevant_set = set(relevant_ids)
    for index, result in enumerate(results, start=1):
        if result.item.id in relevant_set:
            return 1.0 / index
    return 0.0


def citation_accuracy(results: Sequence[RetrievalResult], expected_citation_ids: Iterable[str]) -> float:
    expected_set = set(expected_citation_ids)
    if not results:
        return 0.0
    matched = sum(1 for result in results if result.item.id in expected_set)
    return matched / len(results)


def evaluate_retrieval(
    results: Sequence[RetrievalResult],
    relevant_ids: Iterable[str],
    citation_ids: Iterable[str],
    k: int = 10,
) -> RetrievalEvaluationMetrics:
    precision = precision_at_k(results, relevant_ids, k)
    mrr = mean_reciprocal_rank(results, relevant_ids)
    citation_score = citation_accuracy(results, citation_ids)
    return RetrievalEvaluationMetrics(
        precision_at_k=precision,
        mean_reciprocal_rank=mrr,
        citation_accuracy=citation_score,
    )
