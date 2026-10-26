
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.main import app


client = TestClient(app)


def test_investigate_returns_404_for_missing_incident():
    with patch(
        "src.main.database.get_incident_by_id",
        return_value=None,
    ):
        response = client.post("/incidents/999/investigate")

    assert response.status_code == 404
    assert response.json()["detail"] == "Incident not found"


def test_investigate_returns_result_for_existing_incident():
    incident = {
        "id": 1,
        "title": "Database outage",
        "description": "Database connections are failing",
        "severity": "high",
        "service_name": "database",
        "status": "open",
    }
    expected = {
        "incident_id": 1,
        "model": "qwen2.5:3b",
        "investigation": "Three plausible root causes...",
    }

    with (
        patch(
            "src.main.database.get_incident_by_id",
            return_value=incident,
        ),
        patch(
            "src.main.investigate_incident",
            return_value=expected,
        ),
    ):
        response = client.post("/incidents/1/investigate")

    assert response.status_code == 200
    assert response.json() == expected


def test_investigate_returns_422_for_value_error():
    incident = {
        "id": 1,
        "title": "Database outage",
        "description": "Database connections are failing",
        "severity": "high",
        "service_name": "database",
        "status": "open",
    }

    with (
        patch(
            "src.main.database.get_incident_by_id",
            return_value=incident,
        ),
        patch(
            "src.main.investigate_incident",
            side_effect=ValueError("Invalid incident data"),
        ),
    ):
        response = client.post("/incidents/1/investigate")

    assert response.status_code == 422
    assert response.json()["detail"] == "Invalid incident data"


def test_investigate_returns_503_when_ollama_is_unavailable():
    incident = {
        "id": 1,
        "title": "Database outage",
        "description": "Database connections are failing",
        "severity": "high",
        "service_name": "database",
        "status": "open",
    }

    with (
        patch(
            "src.main.database.get_incident_by_id",
            return_value=incident,
        ),
        patch(
            "src.main.investigate_incident",
            side_effect=RuntimeError(
                "Could not connect to Ollama. Make sure Ollama is running."
            ),
        ),
    ):
        response = client.post("/incidents/1/investigate")

    assert response.status_code == 503

def test_pending_approvals_requires_authentication():
    with TestClient(app) as client:
        response = client.get("/approvals/pending")
        assert response.status_code == 503


def test_get_approval_requires_authentication():
    with TestClient(app) as client:
        response = client.get("/approvals/1")
        assert response.status_code == 503


def test_decision_requires_authentication():
    with TestClient(app) as client:
        response = client.post(
            "/approvals/1/decision",
            json={
                "decision": "approved",
                "decision_note": "",
            },
        )
        assert response.status_code == 503

