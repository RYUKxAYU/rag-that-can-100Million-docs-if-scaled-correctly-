import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional

OrchestrationEventHandler = Callable[["OrchestrationEvent"], Awaitable[None]]


@dataclass(frozen=True)
class OrchestrationEvent:
    name: str
    payload: Dict[str, Any]
    timestamp: float
    sequence_id: int


class EventBus:
    def __init__(self) -> None:
        self._subscribers: Dict[str, List[OrchestrationEventHandler]] = {}
        self._sequence_counter = 0
        self._logger = logging.getLogger("app.event_bus")

    def subscribe(self, event_name: str, handler: OrchestrationEventHandler) -> None:
        self._subscribers.setdefault(event_name, []).append(handler)

    def unsubscribe(self, event_name: str, handler: OrchestrationEventHandler) -> None:
        handlers = self._subscribers.get(event_name)
        if handlers and handler in handlers:
            handlers.remove(handler)

    async def publish(self, event: OrchestrationEvent) -> None:
        handlers = list(self._subscribers.get(event.name, []))
        for handler in handlers:
            try:
                await handler(event)
            except Exception as exc:
                self._logger.exception("Event handler failed", exc_info=exc)

    def create_event(self, name: str, payload: Dict[str, Any]) -> OrchestrationEvent:
        self._sequence_counter += 1
        return OrchestrationEvent(
            name=name,
            payload=payload,
            timestamp=time.monotonic(),
            sequence_id=self._sequence_counter,
        )
