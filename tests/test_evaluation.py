import asyncio
import json
import tempfile
from pathlib import Path

from app.evaluation.benchmark_runner import BenchmarkRunner, BenchmarkResult
from app.evaluation.hallucination_audit import HallucinationAudit
from app.evaluation.metrics_exporter import MetricsExporter
from app.evaluation.retrieval_evaluator import (
    RetrievalEvaluationMetrics,
    citation_accuracy,
    evaluate_retrieval,
    precision_at_k,
)
from app.retrieval.schema import RetrievalItem, RetrievalResult


def _sample_results() -> list[RetrievalResult]:
    items = [
        RetrievalItem(id=f"doc{i}", text=f"Document {i}", metadata={"source": f"doc{i}"})
        for i in range(5)
    ]
    return [
        RetrievalResult(item=item, score=1.0 / (index + 1), rank=index + 1, source="bm25")
        for index, item in enumerate(items)
    ]


def test_precision_at_k_returns_expected_value() -> None:
    results = _sample_results()
    assert precision_at_k(results, ["doc0", "doc2", "doc4"], k=3) == 2 / 3


def test_citation_accuracy_calculates_on_result_ids() -> None:
    results = _sample_results()
    accuracy = citation_accuracy(results, ["doc0", "doc4"])
    assert accuracy == 2 / 5


def test_evaluate_retrieval_returns_metrics_object() -> None:
    results = _sample_results()
    metrics = evaluate_retrieval(results, ["doc0", "doc2"], ["doc0", "doc1"])
    assert isinstance(metrics, RetrievalEvaluationMetrics)
    assert metrics.precision_at_k == 0.4
    assert metrics.mean_reciprocal_rank == 1.0
    assert metrics.citation_accuracy == 0.4


def test_hallucination_audit_flags_unsupported_and_missing_citations() -> None:
    audit = HallucinationAudit()
    result = audit.audit_citations(
        citations=["doc0", "docX"],
        valid_ids=["doc0", "doc1", "doc2"],
        expected_ids=["doc0", "doc1"],
    )
    assert result.total_citations == 2
    assert result.supported_citations == 1
    assert result.unsupported_citations == 1
    assert result.supported_ratio == 0.5
    assert result.missing_expected_citations == ["doc1"]
    assert "unsupported_citations" in result.audit_flags
    assert "missing_expected_citations" in result.audit_flags


def test_metrics_exporter_writes_json_and_csv_files() -> None:
    exporter = MetricsExporter()
    record = {"precision_at_k": 0.8, "throughput_rps": 12.5}
    with tempfile.TemporaryDirectory() as temp_dir:
        json_path = Path(temp_dir) / "metrics.json"
        csv_path = Path(temp_dir) / "metrics.csv"
        exporter.export_json(record, str(json_path))
        exporter.export_csv([record], str(csv_path))

        assert json_path.exists()
        assert csv_path.exists()

        loaded = json.loads(json_path.read_text(encoding="utf-8"))
        assert loaded["precision_at_k"] == 0.8

        csv_data = csv_path.read_text(encoding="utf-8")
        assert "precision_at_k" in csv_data
        assert "throughput_rps" in csv_data


def test_benchmark_runner_run_evaluation_produces_expected_metrics() -> None:
    async def retrieval_func() -> list[RetrievalResult]:
        await asyncio.sleep(0)
        return _sample_results()

    runner = BenchmarkRunner(MetricsExporter())
    result = asyncio.run(
        runner.run_evaluation(
            retrieval_func=retrieval_func,
            query="test",
            relevant_ids=["doc0", "doc3"],
            citation_ids=["doc0", "doc1"],
            top_k=3,
            repetitions=2,
        )
    )
    assert isinstance(result, BenchmarkResult)
    assert result.metrics["precision_at_k"] == 1 / 3
    assert result.metrics["citation_accuracy"] == 2 / 5
    assert result.throughput_rps > 0
    assert result.latency_seconds >= 0


def test_benchmark_runner_export_reports_creates_files() -> None:
    async def retrieval_func() -> list[RetrievalResult]:
        await asyncio.sleep(0)
        return _sample_results()

    runner = BenchmarkRunner(MetricsExporter())
    result = asyncio.run(
        runner.run_evaluation(
            retrieval_func=retrieval_func,
            query="test",
            relevant_ids=["doc0"],
            citation_ids=["doc0"],
            top_k=1,
            repetitions=1,
        )
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        json_path = Path(temp_dir) / "report.json"
        csv_path = Path(temp_dir) / "report.csv"
        runner.export_reports(result, str(json_path), str(csv_path))

        assert json_path.exists()
        assert csv_path.exists()
