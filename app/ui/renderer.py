from typing import Dict, List

from app.retrieval.schema import RetrievalResult


def render_response_text(response: str, citations: List[str], metadata: Dict[str, object]) -> str:
    citation_block = "\n".join([f"- {citation}" for citation in citations])
    metadata_block = "\n".join([f"{key}: {value}" for key, value in metadata.items()])
    return "\n".join([
        response.strip(),
        "",
        "Citations:",
        citation_block if citation_block else "None",
        "",
        "Metadata:",
        metadata_block if metadata_block else "None",
    ])


def render_citation_details(candidates: List[RetrievalResult]) -> str:
    blocks = []
    for candidate in candidates:
        blocks.append(
            f"[{candidate.item.id}] {candidate.item.text}\nSource: {candidate.item.metadata.get('source', 'unknown')}"
        )
    return "\n\n".join(blocks)
