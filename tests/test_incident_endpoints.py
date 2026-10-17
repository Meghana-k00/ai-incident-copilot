
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.main import app


client = TestClient(app)


def test_create_incident():
    incident = {
        "id": 1,
        "title": "Database outage",
        "description": "Connections failing",
        "severity": "high",
        "service_name": "database",
        "status": "open",
    }

    with patch(
        "src.main.database.create_incident",
        return_value=incident,
    ) as mock_create:
        response = client.post(
            "/incidents",
            json={
                "title": "Database outage",
                "description": "Connections failing",
                "severity": "high",
                "service_name": "database",
            },
        )

    assert response.status_code == 201
    assert response.json() == incident
    mock_create.assert_called_once()


def test_list_incidents():
    incidents = [
        {
            "id": 1,
            "title": "Database outage",
            "description": "Connections failing",
            "severity": "high",
            "service_name": "database",
            "status": "open",
        }
    ]

    with patch(
        "src.main.database.get_all_incidents",
        return_value=incidents,
    ):
        response = client.get("/incidents")

    assert response.status_code == 200
    assert response.json() == incidents


def test_get_existing_incident():
    incident = {
        "id": 1,
        "title": "Database outage",
        "description": "Connections failing",
        "severity": "high",
        "service_name": "database",
        "status": "open",
    }

    with patch(
        "src.main.database.get_incident_by_id",
        return_value=incident,
    ):
        response = client.get("/incidents/1")

    assert response.status_code == 200
    assert response.json() == incident


def test_update_incident_status():
    incident = {
        "id": 1,
        "title": "Database outage",
        "description": "Connections failing",
        "severity": "high",
        "service_name": "database",
        "status": "resolved",
    }

    with patch(
        "src.main.database.update_incident_status",
        return_value=incident,
    ) as mock_update:
        response = client.patch(
            "/incidents/1/status",
            json={"status": "resolved"},
        )

    assert response.status_code == 200
    assert response.json() == incident
    mock_update.assert_called_once_with(1, "resolved")


def test_delete_existing_incident():
    with patch(
        "src.main.database.delete_incident",
        return_value=True,
    ):
        response = client.delete("/incidents/1")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Incident deleted successfully",
        "id": 1,
    }
