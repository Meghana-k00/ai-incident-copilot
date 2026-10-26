
import hmac
import os

from fastapi import Header, HTTPException


def require_reviewer(
    authorization: str | None = Header(default=None),
) -> dict[str, str]:
    """Authenticate a reviewer using a configured bearer token."""
    expected_token = os.getenv("APPROVAL_API_TOKEN")
    reviewer_id = os.getenv("APPROVAL_REVIEWER_ID")
    reviewer_role = os.getenv("APPROVAL_REVIEWER_ROLE")

    if not expected_token or not reviewer_id or not reviewer_role:
        raise HTTPException(
            status_code=503,
            detail="Approval authentication is not configured.",
        )

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Bearer token required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    supplied_token = authorization.removeprefix("Bearer ")

    if not hmac.compare_digest(supplied_token, expected_token):
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if reviewer_role != "reviewer":
        raise HTTPException(
            status_code=403,
            detail="Reviewer role required.",
        )

    return {
        "reviewer_id": reviewer_id,
        "role": reviewer_role,
    }
