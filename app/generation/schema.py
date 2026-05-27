from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class GenerationOutput:
    text: str
    confidence: float
    metadata: Dict[str, Any]
    fallback: bool = False


@dataclass(frozen=True)
class PromptPayload:
    system: str
    query: str
    context: str
    citations: str

    def render(self) -> str:
        return "\n".join([
            f"<System>{self.system}</System>",
            f"<Context>{self.context}</Context>",
            f"<Citations>{self.citations}</Citations>",
            f"<Query>{self.query}</Query>",
        ])
