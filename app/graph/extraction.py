import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping


@dataclass(frozen=True)
class Entity:
    id: str
    label: str
    type: str = "entity"
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class Relationship:
    subject: str
    predicate: str
    object: str
    confidence: float = 0.85
    metadata: Mapping[str, str] = None


class EntityConsistencyValidator:
    """Canonicalize and validate extracted entities for deterministic graph construction."""

    @staticmethod
    def normalize_label(label: str) -> str:
        return " ".join(label.strip().split()).lower()

    @staticmethod
    def canonical_id(label: str) -> str:
        return EntityConsistencyValidator.normalize_label(label).replace(" ", "_")

    @staticmethod
    def validate_entities(entities: Iterable[Entity]) -> List[Entity]:
        canonical: Dict[str, Entity] = {}
        for entity in entities:
            normalized = EntityConsistencyValidator.normalize_label(entity.label)
            entity_id = EntityConsistencyValidator.canonical_id(entity.label)
            if normalized in canonical:
                existing = canonical[normalized]
                aliases = tuple(sorted(set(existing.aliases + entity.aliases + (entity.label,))))
                canonical[normalized] = Entity(
                    id=existing.id,
                    label=existing.label,
                    type=existing.type,
                    aliases=aliases,
                )
            else:
                canonical[normalized] = Entity(
                    id=entity_id,
                    label=entity.label.strip(),
                    type=entity.type,
                    aliases=entity.aliases,
                )
        return list(canonical.values())


class EntityExtractor:
    """Deterministic entity extraction based on capitalized phrase heuristics."""

    ENTITY_PATTERN = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")

    def extract(self, text: str, entity_type: str = "entity") -> List[Entity]:
        entities: List[Entity] = []
        for match in self.ENTITY_PATTERN.finditer(text):
            label = match.group(1).strip()
            if len(label) < 2:
                continue
            entity_id = EntityConsistencyValidator.canonical_id(label)
            entities.append(Entity(id=entity_id, label=label, type=entity_type, aliases=(label,)))
        return EntityConsistencyValidator.validate_entities(entities)


class RelationshipExtractor:
    """Extract simple deterministic relation triples from text."""

    CAPITALIZED_PHRASE = r"[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*"
    NOUN_PHRASE = rf"(?:the|a|an)\s+{CAPITALIZED_PHRASE}|{CAPITALIZED_PHRASE}"
    PATTERN_MAP = [
        (re.compile(rf"\b({CAPITALIZED_PHRASE})\s+(is a|is an|are a|are an)\s+({NOUN_PHRASE})\b"), "is_a"),
        (re.compile(rf"\b({CAPITALIZED_PHRASE})\s+(causes|affects|influences|leads to|results in)\s+({NOUN_PHRASE})\b"), "related_to"),
        (re.compile(rf"\b({CAPITALIZED_PHRASE})\s+(connects to|connected to)\s+({NOUN_PHRASE})\b"), "connected_to"),
    ]

    def extract(self, text: str) -> List[Relationship]:
        relationships: List[Relationship] = []
        for pattern, predicate in self.PATTERN_MAP:
            for match in pattern.finditer(text):
                subject = match.group(1).strip()
                obj = match.group(3).strip()
                relationships.append(Relationship(subject=subject, predicate=predicate, object=obj))
        return relationships
