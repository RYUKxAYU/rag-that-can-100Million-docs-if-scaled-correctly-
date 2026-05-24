import io
from pathlib import Path
from typing import Optional

from PyPDF2 import PdfReader

from app.ingestion.schema import Document


def parse_pdf(path: Path, document_id: Optional[str] = None) -> Document:
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(pages).strip()
    metadata = {
        "source": str(path),
        "type": "pdf",
        "pages": len(reader.pages),
    }
    return Document(id=document_id or path.stem, text=text, metadata=metadata)


def parse_pdf_bytes(data: bytes, document_id: str = "pdf_document") -> Document:
    reader = PdfReader(io.BytesIO(data))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(pages).strip()
    metadata = {
        "source": document_id,
        "type": "pdf",
        "pages": len(reader.pages),
    }
    return Document(id=document_id, text=text, metadata=metadata)
