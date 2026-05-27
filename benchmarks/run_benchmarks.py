import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluation.benchmark_runner import BenchmarkRunner
from app.evaluation.metrics_exporter import MetricsExporter
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.embeddings import BGEM3EmbeddingService
from app.retrieval.orchestrator import HybridRetrievalOrchestrator
from app.retrieval.vector_store import VectorStore, VectorRetriever
from app.retrieval.schema import RetrievalItem


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval benchmarks and export evaluation reports.")
    parser.add_argument("--output", default="benchmarks/reports", help="Report output directory")
    parser.add_argument("--repetitions", type=int, default=5, help="Number of benchmark repetitions")
    args = parser.parse_args()

    documents = [
        RetrievalItem(id=f"doc{i}", text="semantic retrieval test document" * 20, metadata={"source": f"doc{i}"})
        for i in range(100)
    ]
    embedding_service = BGEM3EmbeddingService(batch_size=8, embedding_dim=64)
    embeddings = await embedding_service.embed_documents([doc.text for doc in documents])
    store = VectorStore(str(Path("./retrieval_vectors.db")), embedding_dim=64)
    store.upsert(documents, embeddings)
    vector_retriever = VectorRetriever(store)
    bm25_retriever = BM25Retriever(documents)
    orchestrator = HybridRetrievalOrchestrator(bm25_retriever, vector_retriever)

    query_text = "semantic retrieval test"
    query_embedding = await embedding_service.embed_query(query_text)
    relevant_ids = {"doc0", "doc1", "doc2", "doc3", "doc4"}
    citation_ids = {"doc0", "doc1", "doc2"}

    async def retrieval_func() -> list[RetrievalItem]:
        return await orchestrator.retrieve(
            query_embedding=query_embedding,
            query_text=query_text,
            top_k=5,
            bm25_k=5,
            vector_k=5,
        )

    benchmark_runner = BenchmarkRunner(MetricsExporter())
    benchmark_result = await benchmark_runner.run_evaluation(
        retrieval_func=retrieval_func,
        query=query_text,
        relevant_ids=relevant_ids,
        citation_ids=citation_ids,
        top_k=5,
        repetitions=args.repetitions,
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "retrieval_benchmark_report.json"
    csv_path = output_dir / "retrieval_benchmark_report.csv"
    benchmark_runner.export_reports(benchmark_result, str(json_path), str(csv_path))

    print("Benchmark completed.")
    print(f"JSON report: {json_path}")
    print(f"CSV report: {csv_path}")


if __name__ == "__main__":
    asyncio.run(main())
