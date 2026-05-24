import hashlib
import math
import re
from typing import List, Optional, Sequence

from app.retrieval.schema import RetrievalResult

_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> List[str]:
    return [token.lower() for token in _TOKEN_PATTERN.findall(text) if token]


def count_tokens(text: str) -> int:
    return len(_tokenize(text))


def normalize_scores(raw_scores: Sequence[float]) -> List[float]:
    if not raw_scores:
        return []

    min_score = min(raw_scores)
    max_score = max(raw_scores)
    if math.isclose(min_score, max_score):
        return [1.0 for _ in raw_scores]

    return [(score - min_score) / (max_score - min_score) for score in raw_scores]


def cross_encoder_score(query: str, text: str) -> float:
    query_tokens = set(_tokenize(query))
    text_tokens = set(_tokenize(text))
    overlap = 0.0
    if query_tokens or text_tokens:
        overlap = len(query_tokens & text_tokens) / max(1, len(query_tokens | text_tokens))

    digest = hashlib.blake2b(f"{query}|{text}".encode("utf-8"), digest_size=8).digest()
    raw_hash = int.from_bytes(digest, "big") / float((1 << 64) - 1)
    return 0.6 * raw_hash + 0.4 * overlap


def filter_by_threshold(
    candidates: Sequence[RetrievalResult], scores: Sequence[float], threshold: float
) -> List[RetrievalResult]:
    return [
        RetrievalResult(item=candidate.item, score=score, rank=0, source="reranker")
        for candidate, score in zip(candidates, scores)
        if score >= threshold
    ]


class ContextSelector:
    def __init__(self, token_budget: int = 512):
        self.token_budget = token_budget

    def select(self, query: str, candidates: Sequence[RetrievalResult]) -> List[RetrievalResult]:
        if self.token_budget <= 0:
            return []

        selected: List[RetrievalResult] = []
        used_tokens = count_tokens(query)

        for result in sorted(candidates, key=lambda item: (-item.score, item.item.id)):
            item_token_count = count_tokens(result.item.text)
            if used_tokens + item_token_count > self.token_budget:
                continue
            selected.append(result)
            used_tokens += item_token_count

        return selected


class CrossEncoderReranker:
    def __init__(
        self,
        token_budget: int = 512,
        threshold: float = 0.35,
        max_candidates: int = 50,
    ):
        self.token_budget = token_budget
        self.threshold = threshold
        self.max_candidates = max_candidates

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        top_k: int = 10,
        threshold: Optional[float] = None,
        token_budget: Optional[int] = None,
    ) -> List[RetrievalResult]:
        if not query or not candidates:
            return []

        threshold = threshold if threshold is not None else self.threshold
        token_budget = token_budget if token_budget is not None else self.token_budget

        candidates = sorted(candidates, key=lambda item: (-item.score, item.item.id))[: self.max_candidates]
        selector = ContextSelector(token_budget=token_budget)
        selected = selector.select(query, candidates)

        raw_scores = [cross_encoder_score(query, result.item.text) for result in selected]
        normalized_scores = normalize_scores(raw_scores)
        filtered = filter_by_threshold(selected, normalized_scores, threshold)

        filtered.sort(key=lambda result: (-result.score, result.item.id))
        return [
            RetrievalResult(item=result.item, score=result.score, rank=index + 1, source=result.source)
            for index, result in enumerate(filtered[:top_k])
        ]
