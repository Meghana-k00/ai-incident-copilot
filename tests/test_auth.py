
import pytest
from fastapi import HTTPException

from src.auth import require_reviewer


@pytest.fixture
def reviewer_env(monkeypatch):
    monkeypatch.setenv("APPROVAL_API_TOKEN", "test-token-123")
    monkeypatch.setenv("APPROVAL_REVIEWER_ID", "reviewer-1")
    monkeypatch.setenv("APPROVAL_REVIEWER_ROLE", "reviewer")


def test_valid_reviewer_token_returns_identity(reviewer_env):
    result = require_reviewer(
        authorization="Bearer test-token-123"
    )

    assert result == {
        "reviewer_id": "reviewer-1",
        "role": "reviewer",
    }


def test_missing_token_returns_401(reviewer_env):
    with pytest.raises(HTTPException) as exc:
        require_reviewer(authorization=None)

    assert exc.value.status_code == 401


def test_invalid_token_returns_401(reviewer_env):
    with pytest.raises(HTTPException) as exc:
        require_reviewer(authorization="Bearer wrong-token")

    assert exc.value.status_code == 401


def test_missing_auth_configuration_returns_503(monkeypatch):
    monkeypatch.delenv("APPROVAL_API_TOKEN", raising=False)
    monkeypatch.delenv("APPROVAL_REVIEWER_ID", raising=False)
    monkeypatch.delenv("APPROVAL_REVIEWER_ROLE", raising=False)

    with pytest.raises(HTTPException) as exc:
        require_reviewer(authorization="Bearer test-token-123")

    assert exc.value.status_code == 503


def test_non_reviewer_role_returns_403(monkeypatch):
    monkeypatch.setenv("APPROVAL_API_TOKEN", "test-token-123")
    monkeypatch.setenv("APPROVAL_REVIEWER_ID", "user-1")
    monkeypatch.setenv("APPROVAL_REVIEWER_ROLE", "viewer")

    with pytest.raises(HTTPException) as exc:
        require_reviewer(authorization="Bearer test-token-123")

    assert exc.value.status_code == 403
