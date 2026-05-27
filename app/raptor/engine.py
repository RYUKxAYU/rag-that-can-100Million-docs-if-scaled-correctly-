from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from app.raptor.clustering import RecursiveClusterBuilder
from app.raptor.schema import ClusterNode, RaptorDocument, RaptorRetrievalResult
from app.raptor.summarization import HierarchicalSummarizer


class RAPTORHierarchicalIndexer:
    """Builds and indexes a hierarchical RAPTOR tree for multi-level retrieval."""

    def __init__(self) -> None:
        self.root: Optional[ClusterNode] = None
        self.nodes_by_level: Dict[int, List[ClusterNode]] = {}
        self.nodes_by_id: Dict[str, ClusterNode] = {}
        self.parent_map: Dict[str, str] = {}

    def build_index(self, documents: Iterable[RaptorDocument], max_depth: int = 3) -> ClusterNode:
        documents_by_id = {document.id: document for document in documents}
        builder = RecursiveClusterBuilder()
        root = builder.cluster_documents(list(documents_by_id.values()), max_depth=max_depth)

        summarizer = HierarchicalSummarizer()
        summarizer.summarize_tree(root, documents_by_id)

        self.root = root
        self.nodes_by_level.clear()
        self.nodes_by_id.clear()
        self.parent_map.clear()
        self._index_tree(root)
        return root

    def retrieve(self, query: str, level: Optional[int] = None, top_k: int = 3) -> List[RaptorRetrievalResult]:
        if self.root is None:
            return []

        candidates = self._nodes_for_level(level)
        query_terms = {term.lower() for term in query.split() if term}
        scored = []
        for node in candidates:
            score = self._score_node(node, query_terms)
            if score > 0:
                scored.append(RaptorRetrievalResult(node=node, score=score, sources=["raptor"]))

        return sorted(scored, key=lambda item: (-item.score, item.node.id))[:top_k]

    def _nodes_for_level(self, level: Optional[int]) -> List[ClusterNode]:
        if level is None:
            return [node for nodes in self.nodes_by_level.values() for node in nodes]
        return self.nodes_by_level.get(level, [])

    def _score_node(self, node: ClusterNode, query_terms: set[str]) -> float:
        text = f"{node.summary} {' '.join(node.document_ids)}".lower()
        return float(sum(text.count(term) for term in query_terms))

    def _index_tree(self, node: ClusterNode) -> None:
        self.nodes_by_level.setdefault(node.level, []).append(node)
        self.nodes_by_id[node.id] = node
        if node.parent_id is not None:
            self.parent_map[node.id] = node.parent_id
        for child in node.children:
            self._index_tree(child)

    def get_node(self, node_id: str) -> Optional[ClusterNode]:
        return self.nodes_by_id.get(node_id)

    def get_level_nodes(self, level: int) -> List[ClusterNode]:
        return list(self.nodes_by_level.get(level, []))

    def get_root(self) -> Optional[ClusterNode]:
        return self.root


class RAPTORTraversalEngine:
    """Provides deterministic tree traversal and explanation for RAPTOR retrieval."""

    def __init__(self, indexer: RAPTORHierarchicalIndexer) -> None:
        self.indexer = indexer

    def traverse(self, query: str, max_depth: int = 3) -> List[ClusterNode]:
        results = self.indexer.retrieve(query, top_k=1)
        if not results:
            return []

        node = results[0].node
        path = [node]
        while node.parent_id is not None and len(path) <= max_depth:
            parent = self.indexer.get_node(node.parent_id)
            if parent is None:
                break
            path.append(parent)
            node = parent
        return list(reversed(path))

    def explain_path(self, node_id: str) -> List[str]:
        path: List[str] = []
        current_id = node_id
        while current_id:
            node = self.indexer.get_node(current_id)
            if node is None:
                break
            path.append(f"{node.id}@level{node.level}")
            current_id = node.parent_id or ""
        return list(reversed(path))
