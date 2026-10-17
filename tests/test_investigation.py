
import pytest

from src.investigation import prepare_investigation


def sample_incident():
    return {
        "id": 1,
        "title": "Database connection failure",
        "description": "The application cannot connect to the database.",
        "severity": "high",
        "service_name": "database-service",
        "status": "open",
    }


def test_prepare_investigation():
    result = prepare_investigation(sample_incident())

    assert result["incident_id"] == 1
    assert result["severity"] == "high"
    assert result["service_name"] == "database-service"
    assert "Database connection failure" in result["investigation_context"]


def test_reject_missing_incident_details():
    incident = sample_incident()
    incident["title"] = ""

    with pytest.raises(ValueError):
        prepare_investigation(incident)


def test_reject_unsupported_severity():
    incident = sample_incident()
    incident["severity"] = "urgent"

    with pytest.raises(ValueError):
        prepare_investigation(incident)

def test_investigate_incident_returns_result(monkeypatch):
    import json
    from src.investigation import investigate_incident

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def read(self):
            return json.dumps({
                "response": "Check database connectivity and credentials."
            }).encode("utf-8")

    def fake_urlopen(request, timeout=120):
        return FakeResponse()

    monkeypatch.setattr(
        "src.investigation.urllib.request.urlopen",
        fake_urlopen,
    )

    result = investigate_incident(sample_incident())

    assert result["incident_id"] == 1
    assert result["investigation"]
    assert "database" in result["investigation"].lower()

def test_investigation_handles_ollama_unavailable(monkeypatch):
    import urllib.error
    from src.investigation import investigate_incident

    def fake_urlopen(request, timeout=120):
        raise urllib.error.URLError("Connection refused")

    monkeypatch.setattr(
        "src.investigation.urllib.request.urlopen",
        fake_urlopen,
    )

    with pytest.raises(RuntimeError, match="Could not connect to Ollama"):
        investigate_incident(sample_incident())


def test_investigation_handles_invalid_json(monkeypatch):
    from src.investigation import investigate_incident

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def read(self):
            return b"not valid JSON"

    monkeypatch.setattr(
        "src.investigation.urllib.request.urlopen",
        lambda request, timeout=120: FakeResponse(),
    )

    with pytest.raises(RuntimeError, match="invalid response"):
        investigate_incident(sample_incident())


def test_investigation_handles_unexpected_response(monkeypatch):
    import json
    from src.investigation import investigate_incident

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def read(self):
            return json.dumps(["unexpected data"]).encode("utf-8")

    monkeypatch.setattr(
        "src.investigation.urllib.request.urlopen",
        lambda request, timeout=120: FakeResponse(),
    )

    with pytest.raises(RuntimeError, match="unexpected response"):
        investigate_incident(sample_incident())


def test_investigation_handles_empty_answer(monkeypatch):
    import json
    from src.investigation import investigate_incident

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def read(self):
            return json.dumps({"response": ""}).encode("utf-8")

    monkeypatch.setattr(
        "src.investigation.urllib.request.urlopen",
        lambda request, timeout=120: FakeResponse(),
    )

    with pytest.raises(RuntimeError, match="empty or invalid investigation"):
        investigate_incident(sample_incident())

def test_investigation_handles_retrieval_failure():
    from src.investigation import investigate_incident

    class FailingRetriever:
        def search(self, query, top_k=3):
            raise RuntimeError("Could not connect to Ollama for embeddings.")

    with pytest.raises(
        RuntimeError,
        match="Could not connect to Ollama for embeddings",
    ):
        investigate_incident(
            sample_incident(),
            retriever=FailingRetriever(),
        )
