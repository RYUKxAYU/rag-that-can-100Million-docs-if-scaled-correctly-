import tempfile
from pathlib import Path

from app.ingestion.chunker import semantic_chunk
from app.ingestion.deduper import MinHashDeduplicator
from app.ingestion.engine import ingest_documents
from app.ingestion.markdown_parser import parse_markdown_text
from app.ingestion.pdf_parser import parse_pdf_bytes
from fpdf import FPDF


def _create_pdf_bytes(text: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=10)
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 8, text)
    return bytes(pdf.output())


def test_markdown_semantic_chunking():
    sample_text = "# Title\nThis is a first paragraph. It contains two sentences.\n\n## Subtitle\nThis is a second paragraph with another thought."
    document = parse_markdown_text(sample_text, document_id="sample")
    chunks = semantic_chunk(document, max_words=10, overlap=2)

    assert len(chunks) >= 2
    assert all(chunk.document_id == "sample" for chunk in chunks)
    assert any("Title" in chunk.text or "Subtitle" in chunk.text for chunk in chunks)


def test_minhash_deduplication():
    chunk_a = type("DummyChunk", (), {"text": "This paragraph explains the same concept in a stable way."})
    chunk_b = type("DummyChunk", (), {"text": "This paragraph explains the same concept in a stable way!"})

    deduper = MinHashDeduplicator(threshold=0.9)
    assert not deduper.is_duplicate(chunk_a)
    assert deduper.is_duplicate(chunk_b)


def test_pdf_parser_and_ingestion_pipeline():
    sample_text = "PDF ingestion text. This is a test sentence. Another sentence follows."
    pdf_bytes = _create_pdf_bytes(sample_text)
    document = parse_pdf_bytes(pdf_bytes, document_id="pdf_test")

    assert document.id == "pdf_test"
    assert "PDF ingestion text" in document.text

    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "sample.pdf"
        path.write_bytes(pdf_bytes)
        chunks = ingest_documents([path], max_words=20, overlap=3)

        assert chunks
        assert all(chunk.document_id == path.stem for chunk in chunks)


def test_ingest_documents_mixed_sources():
    markdown = "# Root\nThe system will parse markdown and pdf into semantically chunked content."
    with tempfile.TemporaryDirectory() as temp_dir:
        md_path = Path(temp_dir) / "doc.md"
        md_path.write_text(markdown, encoding="utf-8")
        pdf_path = Path(temp_dir) / "doc.pdf"
        pdf_path.write_bytes(_create_pdf_bytes(markdown))

        chunks = ingest_documents([md_path, pdf_path], max_words=15, overlap=2)
        assert chunks
        assert len({chunk.text for chunk in chunks}) == len(chunks)
