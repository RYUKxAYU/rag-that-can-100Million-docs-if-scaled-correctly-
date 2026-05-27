from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

from app.graph.builder import GraphBuilder, GraphNode, GraphRelationship


@dataclass(frozen=True)
class TraversalNode:
    node: GraphNode
    distance: int


@dataclass(frozen=True)
class TraversalStep:
    subject: GraphNode
    predicate: str
    object: GraphNode
    distance: int


class GraphTraversalEngine:
    """Deterministic graph traversal engine for knowledge-aware retrieval."""

    def __init__(self, graph_builder: GraphBuilder) -> None:
        self.graph_builder = graph_builder

    def traverse(self, start_entity_ids: Iterable[str], max_depth: int = 3) -> List[TraversalNode]:
        queue = deque()
        visited: Set[str] = set()
        result: List[TraversalNode] = []

        for entity_id in sorted(set(start_entity_ids)):
            if self.graph_builder.get_node(entity_id) is not None:
                queue.append((entity_id, 0))
                visited.add(entity_id)

        while queue:
            current_id, distance = queue.popleft()
            node = self.graph_builder.get_node(current_id)
            if node is None:
                continue
            result.append(TraversalNode(node=node, distance=distance))
            if distance >= max_depth:
                continue
            neighbors = self.graph_builder.neighbors(current_id)
            for neighbor_id, _predicate in sorted(neighbors, key=lambda item: item[0]):
                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)
                queue.append((neighbor_id, distance + 1))
        return result

    def find_paths(self, source_id: str, target_id: str, max_depth: int = 4) -> List[List[str]]:
        if self.graph_builder.get_node(source_id) is None or self.graph_builder.get_node(target_id) is None:
            return []

        queue = deque([[source_id]])
        paths: List[List[str]] = []
        visited: Set[str] = {source_id}

        while queue:
            path = queue.popleft()
            if len(path) > max_depth:
                continue
            current = path[-1]
            for neighbor_id, _predicate in sorted(self.graph_builder.neighbors(current), key=lambda item: item[0]):
                if neighbor_id in path:
                    continue
                new_path = path + [neighbor_id]
                if neighbor_id == target_id:
                    paths.append(new_path)
                else:
                    queue.append(new_path)
        return paths

    def explain_traversal(self, start_entity_ids: Iterable[str], max_depth: int = 3) -> List[TraversalStep]:
        steps: List[TraversalStep] = []
        visited: Set[str] = set(start_entity_ids)
        for source_id in sorted(set(start_entity_ids)):
            for target_id, predicate in sorted(self.graph_builder.neighbors(source_id), key=lambda pair: pair[0]):
                subject_node = self.graph_builder.get_node(source_id)
                object_node = self.graph_builder.get_node(target_id)
                if subject_node is None or object_node is None:
                    continue
                steps.append(TraversalStep(subject=subject_node, predicate=predicate, object=object_node, distance=1))
        return steps
