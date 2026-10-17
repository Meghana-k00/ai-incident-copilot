
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from src.models import IncidentCreate, IncidentResponse, IncidentUpdate
from src import database
from src.investigation import investigate_incident


@asynccontextmanager
async def lifespan(app: FastAPI):
    database.init_db()
    yield


app = FastAPI(
    title="AI Incident Copilot",
    description="An AI-powered assistant for incident investigation.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/")
def home():
    return {
        "message": "AI Incident Copilot API is running!",
        "status": "healthy",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "ai-incident-copilot",
    }


@app.post("/incidents", response_model=IncidentResponse, status_code=201)
def create_incident(incident: IncidentCreate):
    return database.create_incident(
        title=incident.title,
        description=incident.description,
        severity=incident.severity,
        service_name=incident.service_name,
    )


@app.get("/incidents", response_model=list[IncidentResponse])
def list_incidents():
    return database.get_all_incidents()


@app.get("/incidents/{incident_id}", response_model=IncidentResponse)
def get_incident(incident_id: int):
    incident = database.get_incident_by_id(incident_id)

    if incident is None:
        raise HTTPException(
            status_code=404,
            detail="Incident not found",
        )

    return incident


@app.patch("/incidents/{incident_id}/status", response_model=IncidentResponse)
def update_incident(
    incident_id: int,
    update: IncidentUpdate,
):
    updated_incident = database.update_incident_status(
        incident_id,
        update.status,
    )

    if updated_incident is None:
        raise HTTPException(
            status_code=404,
            detail="Incident not found",
        )

    return updated_incident


@app.delete("/incidents/{incident_id}")
def delete_incident_endpoint(incident_id: int):
    deleted = database.delete_incident(incident_id)

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Incident not found",
        )

    return {
        "message": "Incident deleted successfully",
        "id": incident_id,
    }

@app.post("/incidents/{incident_id}/investigate")
def investigate_incident_endpoint(incident_id: int):
    incident = database.get_incident_by_id(incident_id)

    if incident is None:
        raise HTTPException(
            status_code=404,
            detail="Incident not found",
        )

    try:
        result = investigate_incident(incident)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    return result
