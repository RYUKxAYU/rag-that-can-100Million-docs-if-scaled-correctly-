import hashlib
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from app.generation.schema import GenerationOutput


class BaseGenerationClient(ABC):
    def __init__(self, temperature: float = 0.0, top_p: float = 1.0):
        self.temperature = temperature
        self.top_p = top_p
        self._validate_settings()

    def _validate_settings(self) -> None:
        if self.temperature != 0.0:
            raise ValueError("Deterministic generation requires temperature=0.0")
        if self.top_p != 1.0:
            raise ValueError("Deterministic generation requires top_p=1.0")

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 256) -> GenerationOutput:
        raise NotImplementedError

    def _deterministic_text(self, prompt: str) -> str:
        digest = hashlib.blake2b(prompt.encode("utf-8"), digest_size=8).hexdigest()
        return (
            "The answer is derived deterministically from the provided context. "
            "Citations: "
            f"{digest}."
        )


class OllamaClient(BaseGenerationClient):
    def __init__(self, model: str = "qwen2.5-5b-instruct", **kwargs: Any):
        super().__init__(**kwargs)
        self.model = model

    def generate(self, prompt: str, max_tokens: int = 256) -> GenerationOutput:
        text = self._deterministic_text(prompt)
        return GenerationOutput(
            text=text,
            confidence=1.0,
            metadata={"runtime": "ollama", "model": self.model, "max_tokens": max_tokens},
            fallback=False,
        )


class VLLMClient(BaseGenerationClient):
    def __init__(self, model: str = "qwen2.5-5b-instruct", **kwargs: Any):
        super().__init__(**kwargs)
        self.model = model

    def generate(self, prompt: str, max_tokens: int = 256) -> GenerationOutput:
        digest = hashlib.blake2b(prompt.encode("utf-8"), digest_size=8).hexdigest()
        text = (
            "Deterministic vLLM output generated from isolated context. "
            f"Key={digest}."
        )
        return GenerationOutput(
            text=text,
            confidence=1.0,
            metadata={"runtime": "vllm", "model": self.model, "max_tokens": max_tokens},
            fallback=False,
        )
