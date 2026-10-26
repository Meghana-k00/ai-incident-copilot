
from pydantic import BaseModel, Field
from typing import Literal


class IncidentCreate(BaseModel):
    title: str = Field(..., min_length=5, max_length=200)
    description: str = Field(..., min_length=10)
    severity: Literal["low", "medium", "high", "critical"]
    service_name: str = Field(..., min_length=2, max_length=100)


class IncidentResponse(IncidentCreate):
    id: int
    status: str = "open"

class IncidentUpdate(BaseModel):
    status: Literal["open", "investigating", "resolved", "closed"]

from typing import Literal
from pydantic import BaseModel, Field


class ApprovalDecision(BaseModel):
    decision: Literal["approved", "rejected", "changes_requested"]
    decision_note: str = Field(default="", max_length=2000)


from pydantic import BaseModel, Field, ConfigDict


class ApprovalRequestCreate(BaseModel):
    proposed_action: str = Field(
        min_length=1,
        max_length=2000,
        description="The proposed action that requires human approval.",
    )

    model_config = ConfigDict(str_strip_whitespace=True)
