import hashlib
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from app.cache.sqlite_cache import SQLiteCacheStore

CACHE_THRESHOLD = 0.98


@dataclass(frozen=True)
class CacheEntry:
    key: str
    query: str
    response: str
    citations: List[str]
    metadata: Dict[str, Any]
    score: float


class SemanticCache:
    def __init__(self, store: Optional[SQLiteCacheStore] = None, threshold: float = CACHE_THRESHOLD):
        self.store = store or SQLiteCacheStore()
        self.threshold = threshold

    def _query_key(self, query: str) -> str:
        return hashlib.blake2b(query.encode("utf-8"), digest_size=16).hexdigest()

    def _query_similarity(self, query_a: str, query_b: str) -> float:
        return SequenceMatcher(None, query_a, query_b).ratio()

    def get(self, query: str) -> Optional[CacheEntry]:
        best_match: Optional[CacheEntry] = None
        best_score = 0.0
        for entry in self.store.iterator():
            score = self._query_similarity(query, entry["query"])
            if score > best_score:
                best_score = score
                best_match = CacheEntry(
                    key=entry["id"],
                    query=entry["query"],
                    response=entry["response"],
                    citations=entry["citations"],
                    metadata=entry["metadata"],
                    score=score,
                )
        if best_match is None or best_score < self.threshold:
            return None
        return best_match

    def store_response(
        self,
        query: str,
        response: str,
        citations: List[str],
        metadata: Dict[str, Any],
    ) -> CacheEntry:
        key = self._query_key(query)
        self.store.upsert(key, query, response, citations, metadata)
        return CacheEntry(
            key=key,
            query=query,
            response=response,
            citations=citations,
            metadata=metadata,
            score=1.0,
        )

    def close(self) -> None:
        self.store.close()
