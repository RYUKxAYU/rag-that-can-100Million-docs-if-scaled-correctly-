from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict


@dataclass(frozen=True)
class QueryComplexity:
    token_count: int
    clause_count: int
    complexity_score: float
    is_complex: bool


class QueryComplexityEvaluator:
    clause_separators = re.compile(r"[\?,;:]| and | or | then | therefore | because ")

    def evaluate(self, query: str) -> QueryComplexity:
        normalized = str(query).strip()
        if not normalized:
            raise ValueError("Query must be a non-empty string for complexity evaluation.")

        tokens = normalized.split()
        token_count = len(tokens)
        clauses = self.clause_separators.findall(normalized.lower())
        clause_count = max(1, len(clauses) + 1)
        complexity_score = min(1.0, (token_count / 20) + (clause_count * 0.1))
        is_complex = complexity_score >= 0.45

        return QueryComplexity(
            token_count=token_count,
            clause_count=clause_count,
            complexity_score=round(complexity_score, 3),
            is_complex=is_complex,
        )

    def summarize(self, complexity: QueryComplexity) -> Dict[str, object]:
        return {
            "token_count": complexity.token_count,
            "clause_count": complexity.clause_count,
            "complexity_score": complexity.complexity_score,
            "is_complex": complexity.is_complex,
        }
