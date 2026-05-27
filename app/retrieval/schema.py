from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class RetrievalItem:
    id: str
    text: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class RetrievalResult:
    item: RetrievalItem
    score: float
    rank: int
    source: str
