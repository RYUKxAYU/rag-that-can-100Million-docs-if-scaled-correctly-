import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.evaluation.benchmark_runner import BenchmarkRunner
from app.evaluation.metrics_exporter import MetricsExporter
from app.retrieval import (
    BGEM3EmbeddingService,
    BM25Retriever,
    CorrectiveRetrievalManager,
    HybridRetrievalOrchestrator,
    VectorRetriever,
    VectorStore,
)
from app.retrieval.schema import RetrievalItem


def run_crag_benchmark() -> None:
    documents = [
        RetrievalItem(
            id=f"doc{i}",
            text="".join([
                "Supply chain pricing and evidence-based retrieval are critical. "
                if i % 2 == 0 else "Agentic retrieval systems require deterministic evaluation. "
                for _ in range(5)
            ]),
            metadata={"source": f"doc{i}"},
        )
        for i in range(30)
    ]

    embedding_service = BGEM3EmbeddingService(batch_size=4, embedding_dim=64)
    query_text = "supply chain pricing evidence retrieval"

    async def benchmark_flow() -> None:
        embeddings = await embedding_service.embed_documents([doc.text for doc in documents])
        store = VectorStore(str(Path("./benchmark_crag_vectors.db")), embedding_dim=64)
        store.upsert(documents, embeddings)
        vector_retriever = VectorRetriever(store)
        bm25_retriever = BM25Retriever(documents)
        orchestrator = HybridRetrievalOrchestrator(bm25_retriever, vector_retriever)
        manager = CorrectiveRetrievalManager(orchestrator)
        query_embedding = await embedding_service.embed_query(query_text)

        async def retrieval_func() -> list[dict]:
            outcome = await manager.run(query_text, query_embedding, top_k=5)
            return outcome.final_results

        benchmark_runner = BenchmarkRunner(MetricsExporter())
        result = await benchmark_runner.run_evaluation(
            retrieval_func=retrieval_func,
            query=query_text,
            relevant_ids={"doc0", "doc2", "doc4", "doc6", "doc8"},
            citation_ids={"doc0", "doc2", "doc4"},
            top_k=5,
            repetitions=3,
        )

        print("CRAG Retrieval Benchmark")
        print("------------------------")
        print(f"Average latency: {result.metrics['average_latency_seconds']:.4f} seconds")
        print(f"Precision@5: {result.metrics['precision_at_k']:.3f}")
        print(f"Citation accuracy: {result.metrics['citation_accuracy']:.3f}")
        print(f"Final query: {query_text}")
        print("Top results:")
        for idx, item in enumerate(result.raw_results[:3], start=1):
            print(f"{idx}. {item.item.id} score={item.score:.3f} source={item.source}")

    asyncio.run(benchmark_flow())


if __name__ == "__main__":
    run_crag_benchmark()
