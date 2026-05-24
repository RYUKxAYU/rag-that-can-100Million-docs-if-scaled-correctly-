import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field


class AppSettings(BaseModel):
    app_name: str = Field("rag_engine")
    environment: str = Field("production")
    log_level: str = Field("INFO")
    host: str = Field("0.0.0.0")
    port: int = Field(8000)
    readiness_timeout_seconds: int = Field(5)
    service_version: str = Field("1.0.0")
    database_dsn: Optional[str] = Field(None)

    model_config = ConfigDict(extra="forbid")

    def __init__(self, **data) -> None:
        env_path = Path(".env")
        if env_path.exists():
            load_dotenv(env_path)

        if not data:
            data = {
                "app_name": os.getenv("APP_NAME", "rag_engine"),
                "environment": os.getenv("APP_ENV", "production"),
                "log_level": os.getenv("LOG_LEVEL", "INFO"),
                "host": os.getenv("APP_HOST", "0.0.0.0"),
                "port": int(os.getenv("APP_PORT", "8000")),
                "readiness_timeout_seconds": int(os.getenv("READINESS_TIMEOUT_SECONDS", "5")),
                "service_version": os.getenv("SERVICE_VERSION", "1.0.0"),
                "database_dsn": os.getenv("DB_DSN"),
            }

        super().__init__(**data)


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()
