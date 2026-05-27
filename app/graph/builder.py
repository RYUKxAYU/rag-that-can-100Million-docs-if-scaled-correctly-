from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import networkx as nx
except ImportError:  # pragma: no cover
    nx = None

from app.graph.extraction import Entity, Relationship


@dataclass(frozen=True)
class GraphNode:
    id: str
    label: str
    type: str = "entity"
    metadata: Dict[str, str] = None


@dataclass(frozen=True)
class GraphRelationship:
    subject: str
    predicate: str
    object: str
    confidence: float = 0.85


class InMemoryGraph:
    def __init__(self) -> None:
        self._nodes: Dict[str, GraphNode] = {}
        self._edges: Dict[str, List[Tuple[str, str]]] = {}

    def add_node(self, node: GraphNode) -> None:
        self._nodes[node.id] = node
        self._edges.setdefault(node.id, [])

    def add_edge(self, subject_id: str, object_id: str, predicate: str) -> None:
        self._edges.setdefault(subject_id, []).append((object_id, predicate))
        self._edges.setdefault(object_id, [])

    def nodes(self) -> List[GraphNode]:
        return list(self._nodes.values())

    def edges(self) -> List[Tuple[str, str, str]]:
        return [(subject, object_id, predicate) for subject, targets in self._edges.items() for object_id, predicate in targets]

    def neighbors(self, node_id: str) -> List[Tuple[str, str]]:
        return sorted(self._edges.get(node_id, []), key=lambda pair: (pair[0], pair[1]))

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        return self._nodes.get(node_id)


class GraphBuilder:
    """Builds graphs deterministically using NetworkX when available or a fallback in-memory graph."""

    def __init__(self, backend: str = "auto") -> None:
        if backend == "networkx" and nx is None:
            raise ImportError("NetworkX backend requested but networkx is not installed.")
        self.backend_name = backend
        self.backend = self._create_backend(backend)
        self._node_index: Dict[str, GraphNode] = {}

    def _create_backend(self, backend: str) -> object:
        if backend == "networkx" and nx is not None:
            return nx.DiGraph()
        return InMemoryGraph()

    def add_entity(self, entity: Entity) -> GraphNode:
        if entity.id in self._node_index:
            return self._node_index[entity.id]
        node = GraphNode(id=entity.id, label=entity.label, type=entity.type)
        self._node_index[entity.id] = node
        if isinstance(self.backend, InMemoryGraph):
            self.backend.add_node(node)
        else:
            self.backend.add_node(entity.id, label=entity.label, type=entity.type)
        return node

    def add_relationship(self, relationship: Relationship) -> GraphRelationship:
        subject_id = EntityConsistencyValidator.canonical_id(relationship.subject)
        object_id = EntityConsistencyValidator.canonical_id(relationship.object)
        if subject_id not in self._node_index:
            self.add_entity(Entity(id=subject_id, label=relationship.subject))
        if object_id not in self._node_index:
            self.add_entity(Entity(id=object_id, label=relationship.object))
        if isinstance(self.backend, InMemoryGraph):
            self.backend.add_edge(subject_id, object_id, relationship.predicate)
        else:
            self.backend.add_edge(subject_id, object_id, predicate=relationship.predicate)
        return GraphRelationship(subject=subject_id, predicate=relationship.predicate, object=object_id, confidence=relationship.confidence)

    def nodes(self) -> List[GraphNode]:
        if isinstance(self.backend, InMemoryGraph):
            return self.backend.nodes()
        return [GraphNode(id=node, label=self.backend.nodes[node].get("label", node), type=self.backend.nodes[node].get("type", "entity")) for node in sorted(self.backend.nodes)]

    def edges(self) -> List[Tuple[str, str, str]]:
        if isinstance(self.backend, InMemoryGraph):
            return self.backend.edges()
        return [(subject, object_id, data.get("predicate", "")) for subject, object_id, data in sorted(self.backend.edges(data=True))]

    def neighbors(self, node_id: str) -> List[Tuple[str, str]]:
        if isinstance(self.backend, InMemoryGraph):
            return self.backend.neighbors(node_id)
        return sorted([(target, data.get("predicate", "")) for _, target, data in self.backend.edges(node_id, data=True)], key=lambda pair: (pair[0], pair[1]))

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        return self._node_index.get(node_id)


# Avoid circular imports by defining dependency after class definitions.
from app.graph.extraction import EntityConsistencyValidator  # noqa: E402
