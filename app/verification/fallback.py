from typing import Dict

from app.generation.schema import GenerationOutput


class FallbackHandler:
    def __init__(
        self,
        fallback_text: str = "Unable to verify the generated answer with the provided citations.",
    ):
        self.fallback_text = fallback_text

    def create_fallback_output(self) -> GenerationOutput:
        return GenerationOutput(
            text=self.fallback_text,
            confidence=0.0,
            metadata={"verified": False, "reason": "citation_verification_failed"},
            fallback=True,
        )

    def create_filtered_output(self, filtered_text: str, metadata: Dict[str, object]) -> GenerationOutput:
        return GenerationOutput(
            text=filtered_text,
            confidence=1.0,
            metadata={**metadata, "verified": True},
            fallback=False,
        )
