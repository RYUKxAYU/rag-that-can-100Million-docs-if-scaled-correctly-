import json
import math
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

from app.retrieval.schema import RetrievalItem, RetrievalResult


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class VectorStore:
    def __init__(self, path: str, embedding_dim: int = 1536):
        self.path = Path(path)
        self.embedding_dim = embedding_dim
        self.connection = sqlite3.connect(str(self.path), check_same_thread=False)
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS vectors (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                metadata TEXT NOT NULL,
                embedding TEXT NOT NULL
            )
            """
        )
        self.connection.commit()

    def upsert(self, items: Sequence[RetrievalItem], embeddings: Sequence[Sequence[float]]) -> None:
        if len(items) != len(embeddings):
            raise ValueError("The number of embeddings must match the number of items.")
        cursor = self.connection.cursor()
        for item, embedding in zip(items, embeddings):
            if len(embedding) != self.embedding_dim:
                raise ValueError("Embedding length does not match embedding_dim.")
            cursor.execute(
                "INSERT OR REPLACE INTO vectors (id, text, metadata, embedding) VALUES (?, ?, ?, ?)",
                (
                    item.id,
                    item.text,
                    json.dumps(dict(item.metadata)),
                    json.dumps(list(embedding)),
                ),
            )
        self.connection.commit()

    def _iterate_vectors(self) -> Iterator[RetrievalItem]:
        cursor = self.connection.cursor()
        for row in cursor.execute("SELECT id, text, metadata, embedding FROM vectors"):
            metadata = json.loads(row[2])
            text = row[1]
            yield RetrievalItem(id=row[0], text=text, metadata=metadata)

    def query(self, query_embedding: Sequence[float], top_k: int = 10) -> List[RetrievalResult]:
        if len(query_embedding) != self.embedding_dim:
            raise ValueError("Query embedding length does not match embedding_dim.")
        cursor = self.connection.cursor()
        results: List[RetrievalResult] = []
        for row in cursor.execute("SELECT id, text, metadata, embedding FROM vectors"):
            embedding = json.loads(row[3])
            similarity = _cosine_similarity(query_embedding, embedding)
            if similarity > 0:
                item = RetrievalItem(id=row[0], text=row[1], metadata=json.loads(row[2]))
                results.append(RetrievalResult(item=item, score=similarity, rank=0, source="vector"))
        results.sort(key=lambda result: (-result.score, result.item.id))
        for index, result in enumerate(results[:top_k], start=1):
            results[index - 1] = RetrievalResult(
                item=result.item,
                score=result.score,
                rank=index,
                source=result.source,
            )
        return results

    def close(self) -> None:
        if self.connection:
            self.connection.close()

    def __enter__(self) -> "VectorStore":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class VectorRetriever:
    def __init__(self, store: VectorStore):
        self.store = store

    async def retrieve(self, query_embedding: Sequence[float], top_k: int = 10) -> List[RetrievalResult]:
        return self.store.query(query_embedding, top_k=top_k)
