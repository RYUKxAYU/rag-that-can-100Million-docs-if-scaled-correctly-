import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional

PipelineHandler = Callable[[Any, Dict[str, Any]], Awaitable[Any]]


@dataclass(frozen=True)
class PipelineStage:
    name: str
    handler: PipelineHandler


@dataclass
class PipelineResult:
    success: bool
    payload: Any
    stage: Optional[str] = None
    error: Optional[str] = None


class PipelineManager:
    def __init__(self) -> None:
        self._stages: List[PipelineStage] = []
        self._logger = logging.getLogger("app.pipeline")

    def add_stage(self, stage: PipelineStage) -> None:
        self._stages.append(stage)

    def clear(self) -> None:
        self._stages.clear()

    async def execute(self, initial_payload: Any, context: Optional[Dict[str, Any]] = None) -> PipelineResult:
        payload = initial_payload
        context = context or {}

        for stage in self._stages:
            try:
                payload = await stage.handler(payload, context)
            except Exception as exc:
                self._logger.exception("Pipeline stage failed", exc_info=exc)
                return PipelineResult(success=False, payload=payload, stage=stage.name, error=str(exc))

        return PipelineResult(success=True, payload=payload)
