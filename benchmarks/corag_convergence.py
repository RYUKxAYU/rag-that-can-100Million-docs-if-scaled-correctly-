import asyncio
import tempfile
from pathlib import Path
from typing import List

from app.retrieval import (
    BM25Retriever,
    BGEM3EmbeddingService,
    HybridRetrievalOrchestrator,
    IterativeRetrievalLoop,
    RecursiveRetrievalOrchestrator,
    VectorStore,
    VectorRetriever,
)
from app.retrieval.schema import RetrievalItem


def _sample_items() -> List[RetrievalItem]:
    return [
        RetrievalItem(id="doc1", text="A quick brown fox jumps over a lazy dog.", metadata={"source": "doc1"}),
        RetrievalItem(id="doc2", text="Supply chain disruption affects pricing and delivery.", metadata={"source": "doc2"}),
        RetrievalItem(id="doc3", text="Iterative retrieval improves coverage for complex queries.", metadata={"source": "doc3"}),
    ]


async def benchmark_convergence() -> None:
    items = _sample_items()
    bm25 = BM25Retriever(items)
    service = BGEM3EmbeddingService(batch_size=1, embedding_dim=16)
    embeddings = await service.embed_documents([item.text for item in items])

    with tempfile.TemporaryDirectory() as temp_dir:
        with VectorStore(str(Path(temp_dir) / "vector.db"), embedding_dim=16) as store:
            store.upsert(items, embeddings)
            vector_retriever = VectorRetriever(store)
            iterative_loop = IterativeRetrievalLoop(
                retriever=HybridRetrievalOrchestrator(bm25, vector_retriever),
                max_iterations=3,
            )
            recursive = RecursiveRetrievalOrchestrator(iterative_loop=iterative_loop, max_depth=1)
            result = await recursive.recursive_retrieve(
                query="Describe the relationship between supply chain and pricing",
                query_embedding=embeddings[0],
                classification="multi-hop",
                top_k=2,
            )

            print("CoRAG Convergence Benchmark")
            print("Root iterations:", result["root"]["iterations"])
            print("Root sufficient:", result["root"]["sufficient"])
            print("Recursive steps:", result["recursive_steps"])
            if result["child"]:
                print("Child iterations:", result["child"]["iterations"])
                print("Child sufficient:", result["child"]["sufficient"])


if __name__ == "__main__":
    asyncio.run(benchmark_convergence())
