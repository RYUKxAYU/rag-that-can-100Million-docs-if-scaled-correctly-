from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence


@dataclass(frozen=True)
class HallucinationAuditResult:
    total_citations: int
    supported_citations: int
    unsupported_citations: int
    supported_ratio: float
    missing_expected_citations: List[str]
    audit_flags: List[str]
    details: Dict[str, Any]


class HallucinationAudit:
    def audit_citations(self, citations: Sequence[str], valid_ids: Iterable[str], expected_ids: Iterable[str]) -> HallucinationAuditResult:
        valid_set = set(valid_ids)
        expected_set = set(expected_ids)
        cited_set = set(citations)

        supported = len([citation for citation in citations if citation in valid_set])
        unsupported = len(citations) - supported
        missing = sorted(list(expected_set - cited_set))

        flags: List[str] = []
        if unsupported > 0:
            flags.append("unsupported_citations")
        if missing:
            flags.append("missing_expected_citations")
        if not citations:
            flags.append("no_citations_provided")

        supported_ratio = supported / max(len(citations), 1)
        details: Dict[str, Any] = {
            "valid_ids": list(valid_set),
            "expected_ids": list(expected_set),
            "cited_ids": list(cited_set),
        }

        return HallucinationAuditResult(
            total_citations=len(citations),
            supported_citations=supported,
            unsupported_citations=unsupported,
            supported_ratio=supported_ratio,
            missing_expected_citations=missing,
            audit_flags=flags,
            details=details,
        )
