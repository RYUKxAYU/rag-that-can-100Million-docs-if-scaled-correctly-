from __future__ import annotations

from typing import List


class RetrievalStrategyRouter:
    def route(self, classification: str, complexity_score: float) -> List[str]:
        if classification == "multi-hop":
            return ["hybrid", "iterative"]

        if classification == "analytical":
            if complexity_score >= 0.55:
                return ["hybrid", "bm25"]
            return ["hybrid"]

        if classification == "factual":
            if complexity_score >= 0.45:
                return ["hybrid"]
            return ["vector"]

        return ["vector"]
