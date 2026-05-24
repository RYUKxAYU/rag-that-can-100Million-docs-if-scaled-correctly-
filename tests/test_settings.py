import os

from app.config import AppSettings


def test_settings_defaults():
    settings = AppSettings()
    assert settings.app_name == "rag_engine"
    assert settings.host == "0.0.0.0"
    assert settings.port == 8000
    assert settings.environment == "production"


def test_settings_with_environment_overrides(monkeypatch):
    monkeypatch.setenv("APP_NAME", "test-app")
    monkeypatch.setenv("APP_PORT", "1234")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    settings = AppSettings()
    assert settings.app_name == "test-app"
    assert settings.port == 1234
    assert settings.log_level == "DEBUG"
