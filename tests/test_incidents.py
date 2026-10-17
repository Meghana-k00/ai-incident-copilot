
import pytest
from fastapi.testclient import TestClient

from src.main import app
from src import database

@pytest.fixture
def client(tmp_path, monkeypatch):
    test_db = tmp_path / "test_incidents.db"
    monkeypatch.setattr(database, "DATABASE_PATH", test_db)

    with TestClient(app) as test_client:
        yield test_client

def sample_incident():
    return {
        "title": "Database connection failure",
        "description": "The application cannot connect to the database.",
        "severity": "high",
        "service_name": "database-service",
    }


def test_health_check(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_create_incident(client):
    response = client.post("/incidents", json=sample_incident())

    assert response.status_code == 201
    assert response.json()["title"] == "Database connection failure"
    assert response.json()["status"] == "open"
    assert "id" in response.json()


def test_list_incidents(client):
    client.post("/incidents", json=sample_incident())

    response = client.get("/incidents")

    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["title"] == "Database connection failure"


def test_get_incident_by_id(client):
    created = client.post(
        "/incidents", json=sample_incident()
    ).json()

    response = client.get(f"/incidents/{created['id']}")

    assert response.status_code == 200
    assert response.json()["id"] == created["id"]


def test_get_nonexistent_incident(client):
    response = client.get("/incidents/999")

    assert response.status_code == 404
    assert response.json()["detail"] == "Incident not found"


def test_update_incident_status(client):
    created_response = client.post(
        "/incidents",
        json=sample_incident(),
    )

    print("CREATE STATUS:", created_response.status_code)
    print("CREATE BODY:", created_response.json())

    assert created_response.status_code == 201

    created = created_response.json()

    response = client.patch(
        f"/incidents/{created['id']}/status",
        json={"status": "investigating"},
    )

    print("UPDATE STATUS:", response.status_code)
    print("UPDATE BODY:", response.json())

    assert response.status_code == 200
    assert response.json()["status"] == "investigating"

def test_reject_invalid_status(client):
    created = client.post(
        "/incidents", json=sample_incident()
    ).json()

    response = client.patch(
        f"/incidents/{created['id']}/status",
        json={"status": "invalid"},
    )

    assert response.status_code == 422


def test_update_nonexistent_incident(client):
    response = client.patch(
        "/incidents/999/status",
        json={"status": "resolved"},
    )

    assert response.status_code == 404


def test_delete_incident(client):
    created = client.post(
        "/incidents", json=sample_incident()
    ).json()

    response = client.delete(f"/incidents/{created['id']}")

    assert response.status_code == 200
    assert response.json()["id"] == created["id"]

    follow_up = client.get(f"/incidents/{created['id']}")
    assert follow_up.status_code == 404


def test_delete_nonexistent_incident(client):
    response = client.delete("/incidents/999")

    assert response.status_code == 404


def test_reject_invalid_incident_data(client):
    response = client.post(
        "/incidents",
        json={
            "title": "Bad",
            "description": "Too short",
            "severity": "urgent",
            "service_name": "x",
        },
    )

    assert response.status_code == 422


def test_empty_incident_list(client):
    response = client.get("/incidents")

    assert response.status_code == 200
    assert response.json() == []


def test_multiple_incidents_have_unique_ids(client):
    first = client.post(
        "/incidents",
        json=sample_incident(),
    )
    second = client.post(
        "/incidents",
        json={
            **sample_incident(),
            "title": "Payment service failure",
        },
    )

    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json()["id"] != second.json()["id"]


def test_updated_status_persists_on_retrieval(client):
    created = client.post(
        "/incidents",
        json=sample_incident(),
    ).json()

    incident_id = created["id"]

    update_response = client.patch(
        f"/incidents/{incident_id}/status",
        json={"status": "resolved"},
    )

    assert update_response.status_code == 200

    get_response = client.get(f"/incidents/{incident_id}")

    assert get_response.status_code == 200
    assert get_response.json()["status"] == "resolved"

def test_investigate_incident_success(client, monkeypatch):
    created = client.post(
        "/incidents",
        json=sample_incident(),
    ).json()

    def fake_investigate(incident, retriever=None):
        return {
            "incident_id": incident["id"],
            "model": "qwen2.5:3b",
            "investigation": "Check database connectivity and server status.",
        }

    monkeypatch.setattr(
        "src.main.investigate_incident",
        fake_investigate,
    )

    response = client.post(
        f"/incidents/{created['id']}/investigate"
    )

    assert response.status_code == 200
    assert response.json()["incident_id"] == created["id"]
    assert response.json()["model"] == "qwen2.5:3b"
    assert "database connectivity" in response.json()["investigation"]


def test_investigate_nonexistent_incident(client, monkeypatch):
    def should_not_run(incident):
        raise AssertionError("Investigation should not run for a missing incident")

    monkeypatch.setattr(
        "src.main.investigate_incident",
        should_not_run,
    )

    response = client.post("/incidents/999/investigate")

    assert response.status_code == 404
    assert response.json()["detail"] == "Incident not found"


def test_investigation_runtime_error(client, monkeypatch):
    created = client.post(
        "/incidents",
        json=sample_incident(),
    ).json()

    def fake_investigate(incident, retriever=None):
        raise RuntimeError("Ollama is unavailable")

    monkeypatch.setattr(
        "src.main.investigate_incident",
        fake_investigate,
    )

    response = client.post(
        f"/incidents/{created['id']}/investigate"
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Ollama is unavailable"