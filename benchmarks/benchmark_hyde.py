import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.evaluation.benchmark_runner import BenchmarkRunner
from app.evaluation.metrics_exporter import MetricsExporter
from app.retrieval import (
    BGEM3EmbeddingService,
    BM25Retriever,
    HyDEQueryExpander,
    HybridRetrievalOrchestrator,
    VectorStore,
    VectorRetriever,
)
from app.retrieval.schema import RetrievalItem


def recall_at_k(results, relevant_ids, k=5):
    relevant_set = set(relevant_ids)
    top_results = results[:k]
    if not top_results:
        return 0.0
    hits = sum(1 for result in top_results if result.item.id in relevant_set)
    return hits / len(relevant_set)


async def run_hyde_benchmark() -> None:
    documents = [
        RetrievalItem(
            id=f"doc{i}",
            text="".join(
                [
                    "Semantic expansion and query recall are key retrieval goals. "
                    if i % 2 == 0
                    else "This document contains unrelated topic noise. "
                    for _ in range(4)
                ]
            ),
            metadata={"source": f"doc{i}"},
        )
        for i in range(20)
    ]

    embedding_service = BGEM3EmbeddingService(batch_size=4, embedding_dim=64)
    bm25_retriever = BM25Retriever(documents)
    embeddings = await embedding_service.embed_documents([doc.text for doc in documents])
    store = VectorStore(str(Path("./benchmark_hyde_vectors.db")), embedding_dim=64)
    store.upsert(documents, embeddings)
    vector_retriever = VectorRetriever(store)
    orchestrator = HybridRetrievalOrchestrator(bm25_retriever, vector_retriever)
    expander = HyDEQueryExpander()

    query_text = "semantic expansion recall"
    query_embedding = await embedding_service.embed_query(query_text)

    async def retrieval_func() -> list:
        outcome = await expander.expand(query_text, orchestrator, top_k=5)
        return outcome.retrieval_results

    benchmark_runner = BenchmarkRunner(MetricsExporter())
    result = await benchmark_runner.run_evaluation(
        retrieval_func=retrieval_func,
        query=query_text,
        relevant_ids={"doc0", "doc2", "doc4", "doc6", "doc8"},
        citation_ids={"doc0", "doc2", "doc4"},
        top_k=5,
        repetitions=3,
    )

    recall_score = recall_at_k(result.raw_results, {"doc0", "doc2", "doc4", "doc6", "doc8"}, k=5)
    print("HyDE Recall Benchmark")
    print("----------------------")
    print(f"Precision@5: {result.metrics['precision_at_k']:.3f}")
    print(f"Mean reciprocal rank: {result.metrics['mean_reciprocal_rank']:.3f}")
    print(f"Recall@5: {recall_score:.3f}")
    print(f"Citation accuracy: {result.metrics['citation_accuracy']:.3f}")


if __name__ == "__main__":
    asyncio.run(run_hyde_benchmark())
