import tempfile
from pathlib import Path

from app.cache.semantic_cache import CACHE_THRESHOLD, SemanticCache
from app.cache.sqlite_cache import SQLiteCacheStore


def test_sqlite_cache_store_persist_and_retrieve():
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "semantic_cache.db"
        with SQLiteCacheStore(str(db_path)) as store:
            store.upsert(
                key="querykey",
                query="What is the quick brown fox?",
                response="The fox is quick.",
                citations=["doc1"],
                metadata={"source": "cache-test"},
            )
        with SQLiteCacheStore(str(db_path)) as store:
            entry = store.get("querykey")
            assert entry is not None
            assert entry["query"] == "What is the quick brown fox?"
            assert entry["response"] == "The fox is quick."
            assert entry["citations"] == ["doc1"]
            assert entry["metadata"]["source"] == "cache-test"


def test_semantic_cache_hits_exact_query():
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "semantic_cache.db"
        with SQLiteCacheStore(str(db_path)) as store:
            cache = SemanticCache(store=store, threshold=CACHE_THRESHOLD)
            cache.store_response(
                query="How fast is the fox?",
                response="The fox is very fast.",
                citations=["doc2"],
                metadata={"source": "semantic-cache"},
            )
            match = cache.get("How fast is the fox?")
            assert match is not None
            assert match.response == "The fox is very fast."
            assert match.citations == ["doc2"]
            assert match.score == 1.0


def test_semantic_cache_rejects_dissimilar_queries_below_threshold():
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "semantic_cache.db"
        with SQLiteCacheStore(str(db_path)) as store:
            cache = SemanticCache(store=store, threshold=CACHE_THRESHOLD)
            cache.store_response(
                query="How fast is the fox?",
                response="The fox is very fast.",
                citations=["doc2"],
                metadata={"source": "semantic-cache"},
            )
            miss = cache.get("What color is the fox?")
            assert miss is None
