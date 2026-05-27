import hashlib
import html
from typing import List, Optional, Sequence

from app.generation.schema import PromptPayload
from app.retrieval.schema import RetrievalResult


def escape_xml(text: str) -> str:
    if text is None:
        return ""
    escaped = html.escape(text, quote=True)
    return escaped.replace("\n", " &#10; ")


class PromptBuilder:
    def __init__(
        self,
        system_instruction: Optional[str] = None,
    ):
        self.system_instruction = system_instruction or (
            "Answer only using the provided context sections. "
            "Do not invent facts or add unsupported claims. "
            "Return a concise deterministic response with citations."
        )

    def build_prompt(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
    ) -> str:
        safe_query = escape_xml(query)
        safe_system = escape_xml(self.system_instruction)
        citations = self._build_citations(candidates)
        context = self._build_context(candidates)

        payload = PromptPayload(
            system=safe_system,
            query=safe_query,
            context=context,
            citations=citations,
        )
        return payload.render()

    def _build_context(self, candidates: Sequence[RetrievalResult]) -> str:
        context_blocks: List[str] = []
        for result in candidates:
            item_text = escape_xml(result.item.text)
            item_id = escape_xml(result.item.id)
            context_blocks.append(
                f"<Source id=\"{item_id}\">{item_text}</Source>"
            )
        return "".join(context_blocks)

    def _build_citations(self, candidates: Sequence[RetrievalResult]) -> str:
        citations = [escape_xml(result.item.id) for result in candidates]
        return ",".join(citations)

    def build_prompt_key(self, query: str, candidates: Sequence[RetrievalResult]) -> str:
        prompt = self.build_prompt(query, candidates)
        return hashlib.blake2b(prompt.encode("utf-8"), digest_size=16).hexdigest()
