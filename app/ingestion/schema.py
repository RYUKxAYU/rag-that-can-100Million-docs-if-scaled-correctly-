from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    document_id: str
    position: int
    metadata: Mapping[str, Any] = field(default_factory=dict)
