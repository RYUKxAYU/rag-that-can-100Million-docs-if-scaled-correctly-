from typing import Sequence

from app.retrieval.schema import RetrievalResult


class ConfidenceGuard:
    def __init__(
        self,
        threshold: float = 0.35,
        fallback_text: str = "Unable to answer with sufficient confidence from the provided context.",
    ):
        self.threshold = threshold
        self.fallback_text = fallback_text

    def is_confident(self, candidates: Sequence[RetrievalResult]) -> bool:
        if not candidates:
            return False
        scores = [candidate.score for candidate in candidates]
        return max(scores) >= self.threshold

    def fallback(self) -> str:
        return self.fallback_text
