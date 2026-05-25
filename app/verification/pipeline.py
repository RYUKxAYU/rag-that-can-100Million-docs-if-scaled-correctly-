from typing import Optional, Sequence

from app.generation.schema import GenerationOutput
from app.retrieval.schema import RetrievalResult
from app.verification.engine import VerificationEngine
from app.verification.fallback import FallbackHandler


class VerificationPipeline:
    def __init__(
        self,
        engine: Optional[VerificationEngine] = None,
        fallback_handler: Optional[FallbackHandler] = None,
    ):
        self.fallback_handler = fallback_handler or FallbackHandler()
        self.engine = engine or VerificationEngine(fallback_handler=self.fallback_handler)

    def run(
        self,
        generated_output: GenerationOutput,
        candidates: Sequence[RetrievalResult],
    ) -> GenerationOutput:
        return self.engine.verify(generated_output, candidates)
from typing import Optional, Sequence

from app.generation.schema import GenerationOutput
from app.retrieval.schema import RetrievalResult
from app.verification.engine import VerificationEngine
from app.verification.fallback import FallbackHandler


class VerificationPipeline:
    def __init__(
        self,
        engine: Optional[VerificationEngine] = None,
        fallback_handler: Optional[FallbackHandler] = None,
    ):
        self.fallback_handler = fallback_handler or FallbackHandler()
        self.engine = engine or VerificationEngine(fallback_handler=self.fallback_handler)

    def run(
        self,
        generated_output: GenerationOutput,
        candidates: Sequence[RetrievalResult],
    ) -> GenerationOutput:
        return self.engine.verify(generated_output, candidates)
