import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


class SQLiteCacheStore:
    def __init__(self, path: str = "semantic_cache.db"):
        self.path = Path(path)
        self.connection = sqlite3.connect(str(self.path), check_same_thread=False)
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                response TEXT NOT NULL,
                citations TEXT NOT NULL,
                metadata TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        self.connection.commit()

    def upsert(
        self,
        key: str,
        query: str,
        response: str,
        citations: List[str],
        metadata: Dict[str, Any],
    ) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO cache (id, query, response, citations, metadata, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                key,
                query,
                response,
                json.dumps(citations),
                json.dumps(metadata),
                time.time(),
            ),
        )
        self.connection.commit()

    def iterator(self) -> Iterator[Dict[str, Any]]:
        cursor = self.connection.cursor()
        for row in cursor.execute("SELECT id, query, response, citations, metadata, created_at FROM cache"):
            yield {
                "id": row[0],
                "query": row[1],
                "response": row[2],
                "citations": json.loads(row[3] or "[]"),
                "metadata": json.loads(row[4] or "{}"),
                "created_at": row[5],
            }

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        cursor = self.connection.cursor()
        row = cursor.execute(
            "SELECT id, query, response, citations, metadata, created_at FROM cache WHERE id = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "query": row[1],
            "response": row[2],
            "citations": json.loads(row[3] or "[]"),
            "metadata": json.loads(row[4] or "{}"),
            "created_at": row[5],
        }

    def close(self) -> None:
        if self.connection:
            self.connection.close()

    def __enter__(self) -> "SQLiteCacheStore":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
