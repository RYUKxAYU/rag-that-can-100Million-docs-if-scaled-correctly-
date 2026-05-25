import asyncio
from fastapi.testclient import TestClient

from app.main import app
from app.orchestration import (
    DependencyInjectionContainer,
    EventBus,
    OrchestrationService,
    PipelineManager,
    PipelineStage,
    PipelineResult,
    Service,
    ServiceManager,
)


class LifecycleRecorder(Service):
    def __init__(self) -> None:
        self.started = False
        self.stopped = False

    async def startup(self) -> None:
        self.started = True

    async def shutdown(self) -> None:
        self.stopped = True


class FailingService(Service):
    async def startup(self) -> None:
        raise RuntimeError("startup failure")

    async def shutdown(self) -> None:
        pass


def test_event_bus_dispatches_in_order_and_continues_after_failure() -> None:
    bus = EventBus()
    order = []

    async def first(event):
        order.append("first")

    async def failing(event):
        raise RuntimeError("handler failed")

    async def second(event):
        order.append("second")

    bus.subscribe("test.event", first)
    bus.subscribe("test.event", failing)
    bus.subscribe("test.event", second)

    event = bus.create_event("test.event", {"value": 1})
    asyncio.run(bus.publish(event))

    assert order == ["first", "second"]


def test_pipeline_manager_executes_stages_in_order_and_reports_failure() -> None:
    manager = PipelineManager()
    execution_order = []

    async def first_stage(payload, context):
        execution_order.append("first")
        context["first"] = True
        return f"{payload}-one"

    async def second_stage(payload, context):
        execution_order.append("second")
        raise ValueError("bad stage")

    manager.add_stage(PipelineStage("first", first_stage))
    manager.add_stage(PipelineStage("second", second_stage))

    result = asyncio.run(manager.execute("start", {}))

    assert execution_order == ["first", "second"]
    assert result.success is False
    assert result.stage == "second"
    assert "bad stage" in result.error


def test_service_manager_starts_and_shuts_down_services_in_reverse_order() -> None:
    service_a = LifecycleRecorder()
    service_b = LifecycleRecorder()
    manager = ServiceManager()
    manager.register_service(service_a)
    manager.register_service(service_b)

    asyncio.run(manager.start_all())
    assert service_a.started is True
    assert service_b.started is True

    asyncio.run(manager.shutdown_all())
    assert service_a.stopped is True
    assert service_b.stopped is True


def test_service_manager_cleans_up_on_startup_failure() -> None:
    service_a = LifecycleRecorder()
    failing_service = FailingService()
    manager = ServiceManager()
    manager.register_service(service_a)
    manager.register_service(failing_service)

    try:
        asyncio.run(manager.start_all())
    except Exception:
        pass

    assert service_a.started is True
    assert service_a.stopped is True


def test_dependency_injection_container_resolves_instances_and_builds_by_annotation() -> None:
    container = DependencyInjectionContainer()
    container.register_singleton("config", {"name": "test"})
    container.register_factory("token", lambda: "secret")
    container.register_factory(str, lambda: "secret")

    assert container.resolve("config")["name"] == "test"
    assert container.resolve("token") == "secret"
    assert container.resolve(str) == "secret"

    class SampleService:
        def __init__(self, config: dict, token: str) -> None:
            self.config = config
            self.token = token

    container.register_singleton(dict, {"name": "test"})
    sample = container.build(SampleService)
    assert sample.config["name"] == "test"
    assert sample.token == "secret"


def test_orchestration_service_runs_default_workflow_and_publishes_events() -> None:
    bus = EventBus()
    service = OrchestrationService(event_bus=bus)
    events = []

    async def capture(event):
        events.append((event.name, event.payload))

    bus.subscribe("workflow.start", capture)
    bus.subscribe("workflow.complete", capture)

    result = asyncio.run(service.run_workflow("Hello World"))

    assert result.success is True
    assert result.payload["query"] == "hello world"
    assert result.payload["validated"] is True
    assert events[0][0] == "workflow.start"
    assert events[1][0] == "workflow.complete"
    assert events[1][1]["success"] is True


def test_application_startup_registers_orchestration_service() -> None:
    with TestClient(app) as client:
        assert hasattr(app.state, "orchestration_service")
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
