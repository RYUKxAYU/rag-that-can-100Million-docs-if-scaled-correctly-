from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Sequence, Set

from app.graph.builder import GraphBuilder
from app.graph.traversal import GraphTraversalEngine
from app.retrieval.schema import RetrievalItem, RetrievalResult


@dataclass(frozen=True)
class GraphRetrievalResult:
    item: RetrievalItem
    score: float
    sources: Sequence[str]


class GraphRetrievalFusion:
    """Combine retrieval responses with graph traversal knowledge in a deterministic way."""

    def __init__(self, graph_builder: GraphBuilder) -> None:
        self.graph_builder = graph_builder
        self.traversal_engine = GraphTraversalEngine(graph_builder)

    def fuse(
        self,
        retrieval_results: Iterable[RetrievalResult],
        focus_entity_ids: Iterable[str],
        boost_factor: float = 0.12,
    ) -> List[GraphRetrievalResult]:
        traversal_nodes = self.traversal_engine.traverse(focus_entity_ids)
        entity_ids = {node.node.id for node in traversal_nodes}
        entity_labels = {node.node.label.lower() for node in traversal_nodes}
        fused_results: List[GraphRetrievalResult] = []

        for result in retrieval_results:
            bonus = 0.0
            text_lower = result.item.text.lower()
            if result.item.id in entity_ids:
                bonus = boost_factor
            elif any(label in text_lower for label in entity_labels):
                bonus = boost_factor / 2
            fused_results.append(
                GraphRetrievalResult(
                    item=result.item,
                    score=result.score + bonus,
                    sources=(result.source, "graph_fusion"),
                )
            )

        return sorted(fused_results, key=lambda r: (-r.score, r.item.id))


class GraphSummarizer:
    """Generate a deterministic summary from graph nodes and relationships."""

    def __init__(self, graph_builder: GraphBuilder) -> None:
        self.graph_builder = graph_builder

    def summarize(self, limit: int = 10) -> str:
        nodes = sorted(self.graph_builder.nodes(), key=lambda node: node.id)[:limit]
        edges = sorted(self.graph_builder.edges(), key=lambda edge: (edge[0], edge[1], edge[2]))[:limit]
        lines: List[str] = [f"Graph summary contains {len(nodes)} nodes and {len(edges)} relationships."]
        for node in nodes:
            lines.append(f"- {node.label} ({node.id})")
        for subject, object_id, predicate in edges:
            lines.append(f"- {subject} --{predicate}--> {object_id}")
        return "\n".join(lines)
