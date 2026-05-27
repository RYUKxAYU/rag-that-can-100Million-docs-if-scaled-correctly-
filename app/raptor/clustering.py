import re
from typing import Iterable, List, Sequence

from app.raptor.schema import ClusterNode, RaptorDocument


class RecursiveClusterBuilder:
    """Builds a deterministic hierarchical cluster tree from documents."""

    def cluster_documents(self, documents: Sequence[RaptorDocument], max_depth: int = 3) -> ClusterNode:
        documents = sorted(documents, key=lambda document: document.id)
        return self._build_cluster(documents, level=0, parent_id=None, max_depth=max_depth)

    def _build_cluster(
        self,
        documents: Sequence[RaptorDocument],
        level: int,
        parent_id: str | None,
        max_depth: int,
    ) -> ClusterNode:
        node_id = self._build_node_id(documents, level)
        citations = {citation for document in documents for citation in sorted(document.citations)}
        node = ClusterNode(
            id=node_id,
            level=level,
            document_ids=[document.id for document in documents],
            citations=citations,
            parent_id=parent_id,
        )

        if len(documents) <= 1 or level >= max_depth:
            return node

        groups = self._partition_documents(documents)
        if len(groups) <= 1:
            return node

        for group in groups:
            if not group:
                continue
            child = self._build_cluster(group, level + 1, node_id, max_depth)
            node.children.append(child)

        return node

    def _partition_documents(self, documents: Sequence[RaptorDocument]) -> List[List[RaptorDocument]]:
        groups = [[], []]
        for document in documents:
            bucket = self._deterministic_bucket(document)
            groups[bucket].append(document)

        if not groups[0] or not groups[1]:
            midpoint = len(documents) // 2
            return [list(documents[:midpoint]), list(documents[midpoint:]) if midpoint else list(documents)]

        return groups

    def _deterministic_bucket(self, document: RaptorDocument) -> int:
        key = self._normalize_text(document.text)
        checksum = sum(ord(ch) for ch in key) if key else sum(ord(ch) for ch in document.id)
        return checksum % 2

    def _normalize_text(self, text: str) -> str:
        return " ".join(re.findall(r"[a-zA-Z0-9]+", text.lower()))

    def _build_node_id(self, documents: Sequence[RaptorDocument], level: int) -> str:
        base = "_".join(sorted(document.id for document in documents))
        return f"raptor_{level}_{base[:64]}"
