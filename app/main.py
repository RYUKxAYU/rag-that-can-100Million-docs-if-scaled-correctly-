import logging
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.health import router as health_router
from app.logging import configure_logging
from app.orchestration import OrchestrationService


def create_app() -> FastAPI:
    configure_logging()
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.service_version,
        description="Enterprise-grade context engine foundation with health checks and structured settings.",
    )

    app.state.start_time = time.monotonic()
    app.state.orchestration_service = OrchestrationService()

    @app.on_event("startup")
    async def startup_event() -> None:
        await app.state.orchestration_service.initialize()

    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        await app.state.orchestration_service.shutdown()

    app.include_router(health_router)

    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next):
        logger = logging.getLogger("app.request")
        logger.info(
            "request.start",
            extra={
                "method": request.method,
                "path": request.url.path,
                "client": request.client.host if request.client else None,
            },
        )
        start = time.monotonic()
        response = await call_next(request)
        duration_ms = int((time.monotonic() - start) * 1000)
        logger.info(
            "request.complete",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        return response

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger = logging.getLogger("app.error")
        logger.exception("Unhandled exception", exc_info=exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal Server Error"},
        )

    @app.get("/", summary="Application root")
    async def root():
        return {"status": "ok", "service": settings.app_name, "version": settings.service_version}

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_config=None,
        access_log=False,
    )
