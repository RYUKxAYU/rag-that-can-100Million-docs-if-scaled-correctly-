import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.retrieval.bm25 import BM25Retriever
from app.retrieval.embeddings import BGEM3EmbeddingService
from app.retrieval.orchestrator import HybridRetrievalOrchestrator
from app.retrieval.vector_store import VectorStore, VectorRetriever
from app.retrieval.schema import RetrievalItem


async def run_benchmark() -> None:
    documents = [
        RetrievalItem(id=f"doc{i}", text=" " .join(["semantic", "retrieval", "test"] * 20), metadata={"source": f"doc{i}"})
        for i in range(50)
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

    start = time.perf_counter()
    results = await orchestrator.retrieve(
        query_embedding=query_embedding,
        query_text=query_text,
        top_k=5,
        bm25_k=5,
        vector_k=5,
    )
    duration = time.perf_counter() - start

    print("Retrieval Benchmark")
    print("-------------------")
    print(f"Retrieved {len(results)} items in {duration:.4f} seconds")
    for result in results:
        print(f"rank={result.rank} id={result.item.id} score={result.score:.4f} source={result.source}")


if __name__ == "__main__":
    asyncio.run(run_benchmark())
