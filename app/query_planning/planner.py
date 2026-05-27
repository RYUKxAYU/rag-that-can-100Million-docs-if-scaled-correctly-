from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List

from app.query_planning.classifier import QueryClassifier
from app.query_planning.evaluator import QueryComplexity, QueryComplexityEvaluator
from app.query_planning.router import RetrievalStrategyRouter


@dataclass(frozen=True)
class QueryPlan:
    classification: str
    complexity: QueryComplexity
    selected_strategies: List[str]


class QueryPlanningEngine:
    def __init__(
        self,
        classifier: QueryClassifier | None = None,
        evaluator: QueryComplexityEvaluator | None = None,
        router: RetrievalStrategyRouter | None = None,
    ) -> None:
        self.classifier = classifier or QueryClassifier()
        self.evaluator = evaluator or QueryComplexityEvaluator()
        self.router = router or RetrievalStrategyRouter()

    def plan(self, query: str) -> QueryPlan:
        classification = self.classifier.classify(query)
        complexity = self.evaluator.evaluate(query)
        selected_strategies = self.router.route(classification, complexity.complexity_score)
        return QueryPlan(
            classification=classification,
            complexity=complexity,
            selected_strategies=selected_strategies,
        )

    def plan_as_dict(self, query: str) -> dict:
        plan = self.plan(query)
        return {
            "classification": plan.classification,
            "complexity": asdict(plan.complexity),
            "selected_strategies": plan.selected_strategies,
        }
