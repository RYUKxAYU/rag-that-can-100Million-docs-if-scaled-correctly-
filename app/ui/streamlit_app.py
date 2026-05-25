from typing import Optional

from app.cache.semantic_cache import SemanticCache
from app.memory.conversational_memory import ConversationalMemory
from app.ui.renderer import render_response_text, render_citation_details


def _import_streamlit():
    try:
        import streamlit as st
    except ModuleNotFoundError as exc:
        raise RuntimeError("Streamlit is required to run the UI. Install streamlit before executing the app.") from exc
    return st


class StreamlitUI:
    def __init__(
        self,
        cache: Optional[SemanticCache] = None,
        memory: Optional[ConversationalMemory] = None,
    ):
        self.cache = cache or SemanticCache()
        self.memory = memory or ConversationalMemory()

    def run(self) -> None:
        st = _import_streamlit()
        st.set_page_config(page_title="Enterprise RAG UI", layout="wide")

        st.title("Enterprise Context Engine")
        user_query = st.text_input("Query", key="query_input")

        if st.button("Search") and user_query:
            cached = self.cache.get(user_query)
            if cached:
                st.success("Cache hit — deterministic response loaded.")
                st.text_area("Response", render_response_text(cached.response, cached.citations, cached.metadata), height=220)
                with st.expander("Citation modal"):
                    if cached.citations:
                        st.write("Citations linked to this cached response:")
                        for citation in cached.citations:
                            st.write(f"- {citation}")
                    else:
                        st.write("No citation details available.")
            else:
                st.warning("Cache miss. No cached response is available for this query.")
                st.text_area("Response", "No deterministic cache entry found.", height=220)

        st.sidebar.header("Conversation Memory")
        st.sidebar.text_area("Recent Memory", self.memory.render_history(), height=220)

    def add_memory(self, user_message: str, assistant_message: str) -> None:
        self.memory.add_turn(user_message, assistant_message)

    def show_cache_entry(self, query: str) -> Optional[dict]:
        match = self.cache.get(query)
        if not match:
            return None
        return {
            "query": match.query,
            "response": match.response,
            "citations": match.citations,
            "metadata": match.metadata,
            "score": match.score,
        }
