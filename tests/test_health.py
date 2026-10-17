
from fastapi.testclient import TestClient

from src.main import app


client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "service": "ai-incident-copilot",
    }


def test_create_incident_rejects_missing_required_fields():
    response = client.post(
        "/incidents",
        json={
            "title": "Database outage",
        },
    )

    assert response.status_code == 422


def test_get_nonexistent_incident_returns_404():
    from unittest.mock import patch

    with patch(
        "src.main.database.get_incident_by_id",
        return_value=None,
    ):
        response = client.get("/incidents/999")

    assert response.status_code == 404
    assert response.json()["detail"] == "Incident not found"
