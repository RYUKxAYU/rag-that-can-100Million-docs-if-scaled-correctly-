from pathlib import Path
import tempfile

from app.cache.semantic_cache import SemanticCache
from app.cache.sqlite_cache import SQLiteCacheStore
from app.memory.conversational_memory import ConversationalMemory
from app.ui.renderer import render_citation_details, render_response_text
from app.ui.streamlit_app import StreamlitUI
from app.retrieval.schema import RetrievalItem, RetrievalResult


def test_render_response_text_includes_citations_and_metadata():
    text = render_response_text(
        response="Answer text.",
        citations=["doc1", "doc2"],
        metadata={"source": "ui-test", "version": 1},
    )
    assert "Answer text." in text
    assert "- doc1" in text
    assert "source: ui-test" in text.lower()
    assert "version: 1" in text


def test_render_citation_details_outputs_source_block():
    items = [
        RetrievalItem(id="doc1", text="The quick fox.", metadata={"source": "doc1"}),
    ]
    results = [RetrievalResult(item=items[0], score=0.9, rank=1, source="bm25")]
    details = render_citation_details(results)
    assert "[doc1] The quick fox." in details
    assert "Source: doc1" in details


def test_streamlit_ui_show_cache_entry_returns_cached_data():
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "semantic_cache.db"
        with SQLiteCacheStore(str(db_path)) as store:
            cache = SemanticCache(store=store)
            cache.store_response(
                query="What is the fox?",
                response="The fox is quick.",
                citations=["doc1"],
                metadata={"source": "cache"},
            )
            ui = StreamlitUI(cache=cache, memory=ConversationalMemory())
            entry = ui.show_cache_entry("What is the fox?")
            assert entry is not None
            assert entry["query"] == "What is the fox?"
            assert entry["response"] == "The fox is quick."
            assert entry["citations"] == ["doc1"]


def test_conversational_memory_retains_only_three_turns():
    memory = ConversationalMemory(max_turns=3)
    memory.add_turn("User 1", "Assistant 1")
    memory.add_turn("User 2", "Assistant 2")
    memory.add_turn("User 3", "Assistant 3")
    memory.add_turn("User 4", "Assistant 4")
    assert len(memory.get_history()) == 3
    assert memory.get_history()[0]["user"] == "User 2"
    assert "User 1" not in memory.render_history()
