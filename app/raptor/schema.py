from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class RaptorDocument:
    id: str
    text: str
    metadata: Dict[str, str] = field(default_factory=dict)
    citations: Set[str] = field(default_factory=set)


@dataclass
class ClusterNode:
    id: str
    level: int
    document_ids: List[str]
    citations: Set[str] = field(default_factory=set)
    summary: str = ""
    children: List["ClusterNode"] = field(default_factory=list)
    parent_id: Optional[str] = None


@dataclass(frozen=True)
class RaptorRetrievalResult:
    node: ClusterNode
    score: float
    sources: List[str]
