import re
from typing import List

from app.ingestion.schema import Chunk, Document

_SENTENCE_PATTERN = re.compile(r"(?<=[\.\!?])\s+")
_HEADING_PATTERN = re.compile(r"(?m)^(#{1,6}\s+.*)$")
_WORD_PATTERN = re.compile(r"\w+", re.UNICODE)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_into_semantic_units(text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []

    sections = _HEADING_PATTERN.split(text)
    units: List[str] = []
    buffer: List[str] = []

    for segment in sections:
        if _HEADING_PATTERN.match(segment):
            if buffer:
                units.extend(_split_sentences(" ".join(buffer)))
                buffer = []
            units.append(segment.strip())
        else:
            buffer.append(segment)

    if buffer:
        units.extend(_split_sentences(" ".join(buffer)))

    return [unit for unit in units if unit]


def _split_sentences(text: str) -> List[str]:
    segments = [segment.strip() for segment in _SENTENCE_PATTERN.split(text) if segment.strip()]
    return segments if segments else [text]


def _count_words(text: str) -> int:
    return len(_WORD_PATTERN.findall(text))


def semantic_chunk(document: Document, max_words: int = 120, overlap: int = 20) -> List[Chunk]:
    units = split_into_semantic_units(document.text)
    chunks: List[Chunk] = []
    accumulated_words: List[str] = []
    position = 0

    def flush_chunk() -> None:
        nonlocal position, accumulated_words
        if not accumulated_words:
            return
        chunk_text = " ".join(accumulated_words).strip()
        chunks.append(
            Chunk(
                id=f"{document.id}_chunk_{position}",
                text=chunk_text,
                document_id=document.id,
                position=position,
                metadata={"source_type": document.metadata.get("type"), "unit_count": len(accumulated_words)},
            )
        )
        position += 1
        overlap_words = accumulated_words[-overlap:] if overlap and len(accumulated_words) > overlap else accumulated_words
        accumulated_words = overlap_words.copy()

    for unit in units:
        token_count = _count_words(unit)
        if token_count >= max_words:
            if accumulated_words:
                flush_chunk()
            words = unit.split()
            for start in range(0, len(words), max_words - overlap):
                segment = words[start : start + max_words]
                chunks.append(
                    Chunk(
                        id=f"{document.id}_chunk_{position}",
                        text=" ".join(segment).strip(),
                        document_id=document.id,
                        position=position,
                        metadata={"source_type": document.metadata.get("type"), "unit_count": len(segment)},
                    )
                )
                position += 1
            accumulated_words = []
            continue

        prospective_length = len(accumulated_words) + token_count
        if prospective_length > max_words and accumulated_words:
            flush_chunk()

        accumulated_words.extend(unit.split())

    if accumulated_words:
        flush_chunk()

    return chunks
