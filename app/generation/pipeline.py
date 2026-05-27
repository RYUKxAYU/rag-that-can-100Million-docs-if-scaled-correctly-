from typing import List, Optional, Sequence

from app.generation.client import BaseGenerationClient
from app.generation.guard import ConfidenceGuard
from app.generation.prompt import PromptBuilder
from app.generation.schema import GenerationOutput
from app.retrieval.schema import RetrievalResult


class InferencePipeline:
    def __init__(
        self,
        client: BaseGenerationClient,
        prompt_builder: Optional[PromptBuilder] = None,
        confidence_guard: Optional[ConfidenceGuard] = None,
        max_tokens: int = 256,
    ):
        self.client = client
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.confidence_guard = confidence_guard or ConfidenceGuard()
        self.max_tokens = max_tokens

    def infer(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        fallback_text: Optional[str] = None,
    ) -> GenerationOutput:
        if not query:
            return GenerationOutput(
                text="No query provided.",
                confidence=0.0,
                metadata={"reason": "missing_query"},
                fallback=True,
            )

        if not self.confidence_guard.is_confident(candidates):
            return GenerationOutput(
                text=fallback_text or self.confidence_guard.fallback(),
                confidence=0.0,
                metadata={"reason": "low_confidence"},
                fallback=True,
            )

        prompt = self.prompt_builder.build_prompt(query, candidates)
        output = self.client.generate(prompt, max_tokens=self.max_tokens)
        return GenerationOutput(
            text=output.text,
            confidence=output.confidence,
            metadata={**output.metadata, "prompt_key": self.prompt_builder.build_prompt_key(query, candidates)},
            fallback=output.fallback,
        )
