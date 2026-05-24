from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from app.config import get_settings


class HealthCheck(BaseModel):
    status: str
    started_at: datetime
    environment: str
    version: str
    checks: Optional[dict] = None


router = APIRouter(prefix="/health", tags=["health"])
settings = get_settings()
startup_at = datetime.now(timezone.utc)


@router.get("/live", response_model=HealthCheck)
def live() -> HealthCheck:
    return HealthCheck(
        status="pass",
        started_at=startup_at,
        environment=settings.environment,
        version=settings.service_version,
        checks={"application": "running"},
    )


@router.get("/ready", response_model=HealthCheck)
def ready() -> HealthCheck:
    return HealthCheck(
        status="pass",
        started_at=startup_at,
        environment=settings.environment,
        version=settings.service_version,
        checks={
            "application": "running",
            "configuration": "loaded",
            "readiness_timeout_seconds": settings.readiness_timeout_seconds,
        },
    )
