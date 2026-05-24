from pathlib import Path
from typing import List, Optional, Sequence

from app.ingestion.chunker import semantic_chunk
from app.ingestion.deduper import MinHashDeduplicator
from app.ingestion.markdown_parser import parse_markdown
from app.ingestion.pdf_parser import parse_pdf
from app.ingestion.schema import Chunk

SUPPORTED_EXTENSIONS = {".md", ".markdown", ".pdf"}


def ingest_documents(
    paths: Sequence[Path],
    max_words: int = 120,
    overlap: int = 20,
    dedupe_threshold: float = 0.85,
) -> List[Chunk]:
    deduplicator = MinHashDeduplicator(threshold=dedupe_threshold)
    chunks: List[Chunk] = []

    for path in paths:
        if path.suffix.lower() == ".pdf":
            document = parse_pdf(path)
        elif path.suffix.lower() in {".md", ".markdown"}:
            document = parse_markdown(path)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")

        for chunk in semantic_chunk(document, max_words=max_words, overlap=overlap):
            if not deduplicator.is_duplicate(chunk):
                chunks.append(chunk)

    return chunks


def discover_sources(directory: Path, extensions: Optional[Sequence[str]] = None) -> List[Path]:
    extensions = set(extensions or SUPPORTED_EXTENSIONS)
    return [path for path in directory.rglob("*") if path.suffix.lower() in extensions]
