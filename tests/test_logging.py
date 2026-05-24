from app.logging import configure_logging


def test_configure_logging_runs_without_error():
    configure_logging()
