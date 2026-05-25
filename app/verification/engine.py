from typing import Dict, List, Optional, Sequence

from app.generation.schema import GenerationOutput
from app.retrieval.schema import RetrievalResult
from app.verification.citation_parser import CitationParser
from app.verification.fallback import FallbackHandler
from app.verification.validator import SentenceValidator


class VerificationEngine:
    def __init__(
        self,
        fallback_handler: Optional[FallbackHandler] = None,
    ):
        self.fallback_handler = fallback_handler or FallbackHandler()

    def verify(
        self,
        generated_output: GenerationOutput,
        candidates: Sequence[RetrievalResult],
    ) -> GenerationOutput:
        if generated_output.fallback:
            return generated_output

        validator = SentenceValidator(candidates)
        valid_sentences, invalid_sentences = validator.validate(generated_output.text)
        cited_ids = CitationParser.extract_citation_tokens(generated_output.text)

        if not valid_sentences or not cited_ids:
            return self.fallback_handler.create_fallback_output()

        hallucinatory_claims = len(invalid_sentences) > 0
        filtered_text = " ".join(valid_sentences).strip()

        if not filtered_text:
            return self.fallback_handler.create_fallback_output()

        metadata: Dict[str, object] = {
            "verified_citations": sorted(set(cited_ids) & validator.candidate_ids),
            "invalid_citations": sorted(set(cited_ids) - validator.candidate_ids),
            "removed_sentences": invalid_sentences,
            "hallucination_filtered": hallucinatory_claims,
        }

        if metadata["invalid_citations"]:
            return self.fallback_handler.create_fallback_output()

        return self.fallback_handler.create_filtered_output(filtered_text, metadata)
