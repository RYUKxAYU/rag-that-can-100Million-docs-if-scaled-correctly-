import re
from typing import List

_CITATION_PATTERNS = [
    re.compile(r"\[([^\]\n]+)\]"),
    re.compile(r"\(cite:([^\)\n]+)\)"),
    re.compile(r"Citations:\s*([A-Za-z0-9_\-, ]+)", flags=re.IGNORECASE),
]


class CitationParser:
    @staticmethod
    def extract_citation_tokens(text: str) -> List[str]:
        citations: List[str] = []
        for pattern in _CITATION_PATTERNS:
            for match in pattern.findall(text or ""):
                if isinstance(match, tuple):
                    raw = " ".join(match)
                else:
                    raw = match
                citations.extend(CitationParser._normalize_raw(raw))
        return sorted({token for token in citations if token})

    @staticmethod
    def _normalize_raw(raw: str) -> List[str]:
        parts = [part.strip() for part in re.split(r"[;,\n]", raw) if part.strip()]
        normalized = [part for part in parts if part]
        return normalized
