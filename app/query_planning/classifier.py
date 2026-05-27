from __future__ import annotations

from dataclasses import dataclass
from typing import List


QueryClassification = str


@dataclass(frozen=True)
class QueryClassifier:
    factual_tokens: List[str] = (
        "what",
        "who",
        "where",
        "when",
        "which",
        "state",
        "define",
    )
    analytical_tokens: List[str] = (
        "why",
        "how",
        "compare",
        "analysis",
        "evaluate",
        "impact",
        "trend",
    )
    multihop_tokens: List[str] = (
        "relationship",
        "connect",
        "link",
        "chain",
        "consequence",
        "multi-step",
        "multi hop",
        "follow-up",
        "then",
    )

    def classify(self, query: str) -> QueryClassification:
        normalized = str(query).lower().strip()
        if not normalized:
            raise ValueError("Query must be a non-empty string for classification.")

        if any(token in normalized for token in self.multihop_tokens):
            return "multi-hop"

        if any(token in normalized for token in self.analytical_tokens):
            return "analytical"

        if any(normalized.startswith(token) for token in self.factual_tokens):
            return "factual"

        if len(normalized.split()) < 5:
            return "factual"

        return "analytical"
