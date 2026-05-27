import re
from typing import Dict, Iterable, List, Set

from app.raptor.schema import ClusterNode, RaptorDocument


class HierarchicalSummarizer:
    """Generates deterministic summaries and validates recursive abstraction."""

    def summarize_tree(self, root: ClusterNode, documents: Dict[str, RaptorDocument]) -> None:
        self._summarize_node(root, documents)

    def validate_abstraction(self, root: ClusterNode) -> bool:
        if not root.children:
            return bool(root.summary)

        child_summaries = [child.summary for child in root.children]
        citation_union = {citation for child in root.children for citation in child.citations}
        if root.citations != citation_union:
            return False
        if len(root.summary) > sum(len(summary) for summary in child_summaries):
            return False

        return all(self.validate_abstraction(child) for child in root.children)

    def _summarize_node(self, node: ClusterNode, documents: Dict[str, RaptorDocument]) -> str:
        if node.children:
            for child in node.children:
                self._summarize_node(child, documents)
            source_texts = [child.summary for child in node.children]
        else:
            source_texts = [documents[document_id].text for document_id in node.document_ids]

        content = self._normalize_content(" ".join(sorted(source_texts)))
        sentences = self._extract_sentences(content)
        summary_text = " ".join(sentences[:2]).strip()
        citations = sorted(node.citations)
        node.summary = (
            f"{summary_text} [citations: {', '.join(citations)}]"
            if citations
            else summary_text
        )
        return node.summary

    def _normalize_content(self, content: str) -> str:
        return " ".join(content.split())

    def _extract_sentences(self, content: str) -> List[str]:
        if not content:
            return []
        sentences = re.split(r"(?<=[.!?])\s+", content)
        return [sentence.strip() for sentence in sentences if sentence.strip()]
