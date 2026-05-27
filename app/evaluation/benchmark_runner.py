import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

from app.evaluation.metrics_exporter import MetricsExporter
from app.evaluation.retrieval_evaluator import RetrievalEvaluationMetrics, evaluate_retrieval
from app.retrieval.schema import RetrievalResult


@dataclass(frozen=True)
class BenchmarkResult:
    metrics: Dict[str, Any]
    latency_seconds: float
    throughput_rps: float
    evaluation: RetrievalEvaluationMetrics
    raw_results: Sequence[RetrievalResult]


class BenchmarkRunner:
    def __init__(self, exporter: Optional[MetricsExporter] = None) -> None:
        self.exporter = exporter or MetricsExporter()

    async def run_latency_benchmark(
        self,
        retrieval_func: Any,
        query: str,
        repetitions: int = 5,
    ) -> Dict[str, Any]:
        latencies: List[float] = []
        for _ in range(repetitions):
            start = time.perf_counter()
            await retrieval_func()
            latencies.append(time.perf_counter() - start)
        return {
            "average_latency_seconds": sum(latencies) / len(latencies),
            "min_latency_seconds": min(latencies),
            "max_latency_seconds": max(latencies),
            "runs": len(latencies),
        }

    async def run_throughput_benchmark(
        self,
        retrieval_func: Any,
        concurrency: int = 1,
        duration_seconds: int = 5,
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()
        completed = 0
        while time.perf_counter() - start_time < duration_seconds:
            if concurrency <= 1:
                await retrieval_func()
            else:
                tasks = [asyncio.create_task(retrieval_func()) for _ in range(concurrency)]
                await asyncio.gather(*tasks)
            completed += concurrency
        total_time = time.perf_counter() - start_time
        return {
            "duration_seconds": round(total_time, 4),
            "completed_requests": completed,
            "throughput_rps": completed / total_time if total_time > 0 else 0.0,
        }

    async def run_evaluation(
        self,
        retrieval_func: Any,
        query: str,
        relevant_ids: Iterable[str],
        citation_ids: Iterable[str],
        top_k: int = 10,
        repetitions: int = 5,
    ) -> BenchmarkResult:
        if repetitions <= 0:
            raise ValueError("Repetitions must be greater than zero.")

        latencies: List[float] = []
        raw_results: List[Sequence[RetrievalResult]] = []
        for _ in range(repetitions):
            start = time.perf_counter()
            results = await retrieval_func()
            latencies.append(time.perf_counter() - start)
            raw_results.append(results)

        average_latency = sum(latencies) / len(latencies)
        throughput = len(latencies) / sum(latencies) if sum(latencies) > 0 else 0.0
        evaluation = evaluate_retrieval(raw_results[-1], relevant_ids, citation_ids, k=top_k)

        metrics = {
            "average_latency_seconds": average_latency,
            "throughput_rps": throughput,
            "precision_at_k": evaluation.precision_at_k,
            "mean_reciprocal_rank": evaluation.mean_reciprocal_rank,
            "citation_accuracy": evaluation.citation_accuracy,
        }

        return BenchmarkResult(
            metrics=metrics,
            latency_seconds=average_latency,
            throughput_rps=throughput,
            evaluation=evaluation,
            raw_results=raw_results[-1],
        )

    def export_reports(self, benchmark_result: BenchmarkResult, json_path: str, csv_path: str) -> None:
        report = {
            "metrics": benchmark_result.metrics,
            "raw_ids": [result.item.id for result in benchmark_result.raw_results],
        }
        self.exporter.export_json(report, json_path)
        self.exporter.export_csv([benchmark_result.metrics], csv_path)
