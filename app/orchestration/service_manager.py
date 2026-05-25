import asyncio
import logging
from abc import ABC, abstractmethod
from typing import List


class ServiceLifecycleError(RuntimeError):
    pass


class Service(ABC):
    @abstractmethod
    async def startup(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def shutdown(self) -> None:
        raise NotImplementedError


class ServiceManager:
    def __init__(self) -> None:
        self._services: List[Service] = []
        self._logger = logging.getLogger("app.service_manager")

    def register_service(self, service: Service) -> None:
        if service not in self._services:
            self._services.append(service)

    async def start_all(self) -> None:
        started: List[Service] = []
        for service in self._services:
            try:
                await service.startup()
                started.append(service)
            except Exception as exc:
                self._logger.exception("Service startup failed", exc_info=exc)
                await self._stop_started(started)
                raise ServiceLifecycleError(f"Failed to start service {service.__class__.__name__}") from exc

    async def shutdown_all(self) -> None:
        for service in reversed(self._services):
            try:
                await service.shutdown()
            except Exception as exc:
                self._logger.exception("Service shutdown failed", exc_info=exc)

    async def _stop_started(self, services: List[Service]) -> None:
        for service in reversed(services):
            try:
                await service.shutdown()
            except Exception as exc:
                self._logger.exception("Failed to shutdown started service", exc_info=exc)
