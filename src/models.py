
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