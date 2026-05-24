from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_live_health_check():
    response = client.get("/health/live")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "pass"
    assert payload["checks"]["application"] == "running"


def test_ready_health_check():
    response = client.get("/health/ready")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "pass"
    assert payload["checks"]["configuration"] == "loaded"
