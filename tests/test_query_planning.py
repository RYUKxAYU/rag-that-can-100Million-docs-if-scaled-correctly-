from app.query_planning import (
    QueryClassifier,
    QueryComplexityEvaluator,
    QueryPlanningEngine,
    RetrievalStrategyRouter,
)


def test_query_classifier_identifies_factual_query() -> None:
    classifier = QueryClassifier()
    assert classifier.classify("What is the capital of France?") == "factual"


def test_query_classifier_identifies_analytical_query() -> None:
    classifier = QueryClassifier()
    assert classifier.classify("Why does this trend occur in the market?") == "analytical"


def test_query_classifier_identifies_multihop_query() -> None:
    classifier = QueryClassifier()
    assert classifier.classify("Explain the relationship between inflation and interest rates.") == "multi-hop"


def test_query_complexity_evaluator_returns_complexity_metadata() -> None:
    evaluator = QueryComplexityEvaluator()
    complexity = evaluator.evaluate("Compare and contrast the two models, then outline the consequences.")

    assert complexity.token_count > 5
    assert complexity.clause_count >= 2
    assert complexity.complexity_score > 0.3
    assert complexity.is_complex is True


def test_retrieval_strategy_router_dynamic_selection() -> None:
    router = RetrievalStrategyRouter()
    assert router.route("factual", 0.2) == ["vector"]
    assert router.route("factual", 0.7) == ["hybrid"]
    assert router.route("analytical", 0.4) == ["hybrid"]
    assert router.route("analytical", 0.6) == ["hybrid", "bm25"]
    assert router.route("multi-hop", 0.2) == ["hybrid", "iterative"]


def test_query_planning_engine_produces_query_plan() -> None:
    planner = QueryPlanningEngine()
    plan = planner.plan("How does supply chain disruption affect pricing?")

    assert plan.classification in {"analytical", "multi-hop"}
    assert plan.complexity.token_count > 0
    assert isinstance(plan.selected_strategies, list)
