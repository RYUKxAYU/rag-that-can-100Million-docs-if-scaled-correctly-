from pathlib import Path
from typing import Optional

from app.ingestion.schema import Document


def parse_markdown(path: Path, document_id: Optional[str] = None) -> Document:
    text = path.read_text(encoding="utf-8").strip()
    metadata = {
        "source": str(path),
        "type": "markdown",
    }
    return Document(id=document_id or path.stem, text=text, metadata=metadata)


def parse_markdown_text(text: str, document_id: str = "markdown_document") -> Document:
    normalized = text.strip()
    metadata = {
        "source": document_id,
        "type": "markdown",
    }
    return Document(id=document_id, text=normalized, metadata=metadata)
