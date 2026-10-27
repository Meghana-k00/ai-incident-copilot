
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



def test_decision_resume_failure_leaves_approval_pending():
    approval = {
        "id": 1,
        "status": "pending",
        "workflow_thread_id": "test-thread",
    }

    with (
        patch.dict(
            "os.environ",
            {
                "APPROVAL_API_TOKEN": "test-token",
                "APPROVAL_REVIEWER_ID": "reviewer-test",
                "APPROVAL_REVIEWER_ROLE": "reviewer",
            },
        ),
        patch(
            "src.main.database.get_approval_request",
            return_value=approval,
        ),
        patch(
            "src.main.database.claim_approval_decision",
            return_value={
                "outcome": "created",
                "claim": {
                    "approval_id": 1,
                    "decision": "approved",
                    "reviewer_id": "reviewer-test",
                    "decision_note": "Test",
                    "state": "claimed",
                },
            },
        ) as claim_mock,
        patch(
            "src.main.resume_approval",
            side_effect=RuntimeError("Simulated workflow failure"),
        ) as resume_mock,
        patch(
            "src.main.database.decide_approval_request",
        ) as decide_mock,
        TestClient(app) as test_client,
    ):
        response = test_client.post(
            "/approvals/1/decision",
            headers={"Authorization": "Bearer test-token"},
            json={
                "decision": "approved",
                "decision_note": "Test",
            },
        )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Could not resume the approval workflow. "
        "Retry using the same decision and review details."
    )
    claim_mock.assert_called_once()
    resume_mock.assert_called_once_with(
        app.state.approval_graph,
        "test-thread",
        "approved",
    )
    decide_mock.assert_not_called()
    assert approval["status"] == "pending"



def test_decision_database_update_failure_is_not_hidden():
    approval = {
        "id": 1,
        "status": "pending",
        "workflow_thread_id": "test-thread",
    }

    with (
        patch.dict(
            "os.environ",
            {
                "APPROVAL_API_TOKEN": "test-token",
                "APPROVAL_REVIEWER_ID": "reviewer-test",
                "APPROVAL_REVIEWER_ROLE": "reviewer",
            },
        ),
        patch(
            "src.main.database.get_approval_request",
            return_value=approval,
        ),
        patch(
            "src.main.database.claim_approval_decision",
            return_value={
                "outcome": "created",
                "claim": {
                    "approval_id": 1,
                    "decision": "approved",
                    "reviewer_id": "reviewer-test",
                    "decision_note": "Test",
                    "state": "claimed",
                },
            },
        ) as claim_mock,
        patch(
            "src.main.resume_approval",
            return_value={"decision": "approved"},
        ) as resume_mock,
        patch(
            "src.main.database.mark_approval_workflow_resumed",
            return_value={
                "approval_id": 1,
                "decision": "approved",
                "reviewer_id": "reviewer-test",
                "decision_note": "Test",
                "state": "workflow_resumed",
            },
        ),
        patch(
            "src.main.database.decide_approval_request",
            return_value=None,
        ) as decide_mock,
        TestClient(app) as test_client,
    ):
        response = test_client.post(
            "/approvals/1/decision",
            headers={"Authorization": "Bearer test-token"},
            json={
                "decision": "approved",
                "decision_note": "Test",
            },
        )

    assert response.status_code == 409
    assert response.json()["detail"] == (
        "The workflow resumed, but the approval record could "
        "not be finalized. The saved decision claim is preserved."
    )
    claim_mock.assert_called_once()
    resume_mock.assert_called_once_with(
        app.state.approval_graph,
        "test-thread",
        "approved",
    )
    decide_mock.assert_called_once_with(
        approval_id=1,
        decision="approved",
        reviewer_id="reviewer-test",
        decision_note="Test",
    )

def test_resume_failure_preserves_sqlite_approval_and_audit(
    tmp_path, monkeypatch
):
    from unittest.mock import patch
    from fastapi.testclient import TestClient

    from src import database

    test_db = tmp_path / "approval_failure_test.db"
    monkeypatch.setattr(database, "DATABASE_PATH", test_db)
    database.init_db()

    incident = database.create_incident(
        title="Test database outage",
        description="A test incident for approval failure handling.",
        severity="high",
        service_name="test-database",
    )
    approval = database.create_approval_request(
        incident_id=incident["id"],
        proposed_action="Restart the test database",
        workflow_thread_id="test-resume-failure",
    )

    with (
        patch.dict(
            "os.environ",
            {
                "APPROVAL_API_TOKEN": "test-token",
                "APPROVAL_REVIEWER_ID": "reviewer-test",
                "APPROVAL_REVIEWER_ROLE": "reviewer",
            },
        ),
        patch(
            "src.main.resume_approval",
            side_effect=RuntimeError("Simulated workflow failure"),
        ),
        TestClient(app) as test_client,
    ):
        response = test_client.post(
            f"/approvals/{approval['id']}/decision",
            headers={"Authorization": "Bearer test-token"},
            json={
                "decision": "approved",
                "decision_note": "Test failure",
            },
        )

    assert response.status_code == 503

    saved_approval = database.get_approval_request(approval["id"])
    assert saved_approval["status"] == "pending"
    assert saved_approval["reviewer_id"] is None

    events = database.get_approval_audit_log(approval["id"])
    assert [event["event_type"] for event in events] == ["created"]


def test_retry_after_workflow_resume_state_save_failure(
    monkeypatch,
):
    approval = {
        "id": 1,
        "status": "pending",
        "workflow_thread_id": "test-thread",
    }

    saved_claim = {
        "approval_id": 1,
        "decision": "approved",
        "reviewer_id": "reviewer-test",
        "decision_note": "Test",
        "state": "claimed",
    }

    with (
        patch.dict(
            "os.environ",
            {
                "APPROVAL_API_TOKEN": "test-token",
                "APPROVAL_REVIEWER_ID": "reviewer-test",
                "APPROVAL_REVIEWER_ROLE": "reviewer",
            },
        ),
        patch(
            "src.main.database.get_approval_request",
            return_value=approval,
        ),
        patch(
            "src.main.database.claim_approval_decision",
            return_value={
                "outcome": "created",
                "claim": saved_claim,
            },
        ),
        patch(
            "src.main.resume_approval",
            return_value={"decision": "approved"},
        ) as resume_mock,
        patch(
            "src.main.database.mark_approval_workflow_resumed",
            side_effect=RuntimeError("Simulated state-save failure"),
        ),
        patch(
            "src.main.database.decide_approval_request",
        ) as decide_mock,
        TestClient(app) as test_client,
    ):
        response = test_client.post(
            "/approvals/1/decision",
            headers={"Authorization": "Bearer test-token"},
            json={
                "decision": "approved",
                "decision_note": "Test",
            },
        )

    assert response.status_code == 503
    assert "workflow resumed" in response.json()["detail"].lower()
    resume_mock.assert_called_once()
    decide_mock.assert_not_called()
