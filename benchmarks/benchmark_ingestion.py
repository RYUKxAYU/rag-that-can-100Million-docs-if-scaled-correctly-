import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpdf import FPDF
from app.ingestion.engine import ingest_documents


def _create_pdf_bytes(text: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=10)
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 8, text)
    return bytes(pdf.output())


def run_benchmark() -> None:
    markdown_content = "# Benchmark\n" + "This is a performance test sentence. " * 50
    pdf_bytes = _create_pdf_bytes(markdown_content)

    with tempfile.TemporaryDirectory() as temp_dir:
        md_file = Path(temp_dir) / "benchmark.md"
        pdf_file = Path(temp_dir) / "benchmark.pdf"
        md_file.write_text(markdown_content, encoding="utf-8")
        pdf_file.write_bytes(pdf_bytes)

        start = time.perf_counter()
        chunks = ingest_documents([md_file, pdf_file], max_words=40, overlap=5)
        duration = time.perf_counter() - start

        print("Benchmark Results")
        print("-----------------")
        print(f"Parsed and ingested {len(chunks)} chunks in {duration:.3f} seconds")


if __name__ == "__main__":
    run_benchmark()
