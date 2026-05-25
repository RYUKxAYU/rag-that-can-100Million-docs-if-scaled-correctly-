import logging
import time
from typing import Any, Dict, Optional

from app.orchestration.dependency_injection import DependencyInjectionContainer
from app.orchestration.event_bus import EventBus, OrchestrationEvent
from app.orchestration.pipeline import PipelineManager, PipelineResult, PipelineStage
from app.orchestration.service_manager import Service, ServiceManager


class OrchestrationService(Service):
    def __init__(
        self,
        container: Optional[DependencyInjectionContainer] = None,
        event_bus: Optional[EventBus] = None,
        pipeline_manager: Optional[PipelineManager] = None,
        service_manager: Optional[ServiceManager] = None,
    ) -> None:
        self.container = container or DependencyInjectionContainer()
        self.event_bus = event_bus or EventBus()
        self.pipeline_manager = pipeline_manager or PipelineManager()
        self.service_manager = service_manager or ServiceManager()
        self._logger = logging.getLogger("app.orchestration")

        self.container.register_singleton(EventBus, self.event_bus)
        self.container.register_singleton(PipelineManager, self.pipeline_manager)
        self.container.register_singleton(ServiceManager, self.service_manager)

        self._configure_default_pipeline()

    async def startup(self) -> None:
        self._logger.info("Orchestration service starting.")

    async def shutdown(self) -> None:
        self._logger.info("Orchestration service stopping.")
        await self.service_manager.shutdown_all()

    async def initialize(self) -> None:
        await self.service_manager.start_all()

    async def run_workflow(self, query: str) -> PipelineResult:
        event = self._create_event("workflow.start", {"query": query})
        await self.event_bus.publish(event)

        result = await self.pipeline_manager.execute(query, {"query": query})
        completion_event = self._create_event(
            "workflow.complete",
            {"query": query, "success": result.success, "stage": result.stage, "error": result.error},
        )
        await self.event_bus.publish(completion_event)
        return result

    def _create_event(self, name: str, payload: Dict[str, Any]) -> OrchestrationEvent:
        return self.event_bus.create_event(name, payload)

    def _configure_default_pipeline(self) -> None:
        self.pipeline_manager.clear()
        self.pipeline_manager.add_stage(PipelineStage("validate_query", self._validate_query))
        self.pipeline_manager.add_stage(PipelineStage("normalize_query", self._normalize_query))
        self.pipeline_manager.add_stage(PipelineStage("finalize_response", self._finalize_response))

    async def _validate_query(self, payload: Any, context: Dict[str, Any]) -> Any:
        if not isinstance(payload, str) or not payload.strip():
            raise ValueError("Query must be a non-empty string.")
        context["validated"] = True
        return payload.strip()

    async def _normalize_query(self, payload: Any, context: Dict[str, Any]) -> Any:
        normalized = str(payload).lower()
        context["normalized"] = normalized
        return normalized

    async def _finalize_response(self, payload: Any, context: Dict[str, Any]) -> Any:
        response = {
            "query": payload,
            "validated": context.get("validated", False),
            "metadata": {"source": "orchestration", "sequence_id": int(time.time())},
        }
        return response
