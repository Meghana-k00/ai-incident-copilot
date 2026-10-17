
from src.investigation import prepare_investigation
from unittest.mock import MagicMock, patch
from src.investigation import investigate_incident


def test_investigation_context_contains_incident_details():
    incident = {
        "id": 1,
        "title": "Database connection timeout",
        "description": "Requests fail when connecting to the database.",
        "severity": "high",
        "service_name": "customer-api",
        "status": "open",
    }

    result = prepare_investigation(incident)

    context = result["investigation_context"]

    assert "Database connection timeout" in context
    assert "customer-api" in context
    assert "high" in context
    assert "database" in context.lower()


def test_investigation_context_includes_current_status():
    incident = {
        "id": 2,
        "title": "API latency",
        "description": "Response times are increasing.",
        "severity": "medium",
        "service_name": "orders-api",
        "status": "investigating",
    }

    result = prepare_investigation(incident)

    assert result["status"] == "investigating"
    assert "investigating" in result["investigation_context"]


def test_investigation_rejects_missing_description():
    incident = {
        "id": 3,
        "title": "Service unavailable",
        "description": "",
        "severity": "critical",
        "service_name": "payment-api",
        "status": "open",
    }

    try:
        prepare_investigation(incident)
    except ValueError as exc:
        assert "required" in str(exc).lower()
    else:
        raise AssertionError(
            "Expected ValueError for a missing description"
        )


def test_investigation_rejects_invalid_severity():
    incident = {
        "id": 4,
        "title": "Application error",
        "description": "The application is failing.",
        "severity": "extreme",
        "service_name": "customer-api",
        "status": "open",
    }

    try:
        prepare_investigation(incident)
    except ValueError as exc:
        assert "severity" in str(exc).lower()
    else:
        raise AssertionError(
            "Expected ValueError for invalid severity"
        )

def test_investigation_returns_ai_response():
    incident = {
        "id": 1,
        "title": "Database connection timeout",
        "description": "Requests are failing.",
        "severity": "high",
        "service_name": "customer-api",
        "status": "open",
    }

    mock_response = MagicMock()
    mock_response.__enter__.return_value.read.return_value = (
        b'{"response": "Check database connectivity and logs."}'
    )

    with patch(
        "src.investigation.urllib.request.urlopen",
        return_value=mock_response,
    ):
        result = investigate_incident(incident)

    assert result["incident_id"] == 1
    assert result["model"] == "qwen2.5:3b"
    assert "database connectivity" in result["investigation"].lower()
    assert result["retrieved_documents"] == []


def test_investigation_uses_retrieved_knowledge():
    incident = {
        "id": 2,
        "title": "Database connection timeout",
        "description": "Database requests are failing.",
        "severity": "high",
        "service_name": "customer-api",
        "status": "open",
    }

    historical_documents = [
        {
            "text": "Previous incidents were caused by connection pool exhaustion.",
            "metadata": {"source": "database_incidents.md"},
        }
    ]

    mock_retriever = MagicMock()
    mock_retriever.search.return_value = historical_documents

    mock_response = MagicMock()
    mock_response.__enter__.return_value.read.return_value = (
        b'{"response": "Check connection pool usage and database logs."}'
    )

    with patch(
        "src.investigation.urllib.request.urlopen",
        return_value=mock_response,
    ):
        result = investigate_incident(
            incident,
            retriever=mock_retriever,
        )

    mock_retriever.search.assert_called_once()
    assert result["retrieved_documents"] == historical_documents
    assert "connection pool exhaustion" in (
        result["retrieved_documents"][0]["text"].lower()
    )

def test_investigation_rejects_missing_title():
    incident = {
        "id": 10,
        "title": "",
        "description": "Database requests are failing.",
        "severity": "high",
        "service_name": "customer-api",
        "status": "open",
    }

    try:
        investigate_incident(incident)
    except ValueError as exc:
        assert "required" in str(exc).lower()
    else:
        raise AssertionError(
            "Expected ValueError for a missing title"
        )


def test_investigation_rejects_missing_service_name():
    incident = {
        "id": 11,
        "title": "API failure",
        "description": "Requests are failing.",
        "severity": "critical",
        "service_name": "",
        "status": "open",
    }

    try:
        investigate_incident(incident)
    except ValueError as exc:
        assert "required" in str(exc).lower()
    else:
        raise AssertionError(
            "Expected ValueError for a missing service name"
        )


def test_investigation_rejects_invalid_ollama_response():
    incident = {
        "id": 12,
        "title": "Database timeout",
        "description": "Database requests are failing.",
        "severity": "high",
        "service_name": "customer-api",
        "status": "open",
    }

    mock_response = MagicMock()
    mock_response.__enter__.return_value.read.return_value = (
        b'{"unexpected": "missing response field"}'
    )

    with patch(
        "src.investigation.urllib.request.urlopen",
        return_value=mock_response,
    ):
        try:
            investigate_incident(incident)
        except RuntimeError as exc:
            assert "empty or invalid" in str(exc).lower()
        else:
            raise AssertionError(
                "Expected RuntimeError for an invalid Ollama response"
            )

def test_investigation_handles_ollama_connection_failure():
    incident = {
        "id": 13,
        "title": "Database connection timeout",
        "description": "Requests are failing to connect to the database.",
        "severity": "high",
        "service_name": "customer-api",
        "status": "open",
    }

    with patch(
        "src.investigation.urllib.request.urlopen",
        side_effect=OSError("Ollama connection failed"),
    ):
        try:
            investigate_incident(incident)
        except (RuntimeError, OSError):
            pass
        else:
            raise AssertionError(
                "Expected an error when Ollama is unavailable"
            )
