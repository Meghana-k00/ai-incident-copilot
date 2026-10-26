
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from src.models import IncidentCreate, IncidentResponse, IncidentUpdate
from src import database
from src.investigation import investigate_incident
from src.retrieval import IncidentRetriever
from fastapi import Depends, HTTPException
from src.auth import require_reviewer
from src.models import ApprovalDecision
import os
from langgraph.checkpoint.sqlite import SqliteSaver
from src.approval_workflow import build_approval_graph
import uuid
from src.models import ApprovalRequestCreate

from src.approval_workflow import resume_approval



@asynccontextmanager
async def lifespan(app: FastAPI):
    database.init_db()
    app.state.retriever = IncidentRetriever()

    checkpoint_db = os.getenv(
        "APPROVAL_CHECKPOINT_DB",
        str(database.DATA_DIR / "approval_checkpoints.sqlite"),
    )

    with SqliteSaver.from_conn_string(checkpoint_db) as checkpointer:
        checkpointer.setup()
        app.state.approval_graph = build_approval_graph(checkpointer)
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
        result = investigate_incident(
            incident,
            retriever=getattr(app.state, "retriever", None),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc


    return result

@app.get("/approvals/pending")
def read_pending_approvals(
    reviewer: dict = Depends(require_reviewer),
):
    return database.list_pending_approvals()



@app.get("/approvals/{approval_id}")
def read_approval(
    approval_id: int,
    reviewer: dict = Depends(require_reviewer),
):
    approval = database.get_approval_request(approval_id)

    if approval is None:
        raise HTTPException(
            status_code=404,
            detail="Approval request not found.",
        )

    return approval


@app.post("/approvals/{approval_id}/decision")
def submit_approval_decision(
    approval_id: int,
    decision: ApprovalDecision,
    reviewer: dict = Depends(require_reviewer),
):
    approval = database.get_approval_request(approval_id)

    if approval is None:
        raise HTTPException(
            status_code=404,
            detail="Approval request not found.",
        )

    if approval["status"] != "pending":
        raise HTTPException(
            status_code=409,
            detail="This approval request has already been decided.",
        )

    thread_id = approval.get("workflow_thread_id")
    if not thread_id:
        raise HTTPException(
            status_code=409,
            detail="This approval request is not linked to a workflow.",
        )

    try:
        workflow_result = resume_approval(
            app.state.approval_graph,
            thread_id,
            decision.decision,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Could not resume the approval workflow.",
        ) from exc

    updated_approval = database.decide_approval_request(
        approval_id=approval_id,
        decision=decision.decision,
        reviewer_id=reviewer["reviewer_id"],
        decision_note=decision.decision_note,
    )

    if updated_approval is None:
        raise HTTPException(
            status_code=409,
            detail="Approval request could not be updated.",
        )

    return {
        "approval": updated_approval,
        "workflow": workflow_result,
        "message": "Decision recorded and workflow resumed.",
    }

@app.post("/incidents/{incident_id}/approval-request")
def create_incident_approval_request(
    incident_id: int,
    request: ApprovalRequestCreate,
):
    incident = database.get_incident_by_id(incident_id)
    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found.")

    thread_id = str(uuid.uuid4())

    approval = database.create_approval_request(
        incident_id=incident_id,
        proposed_action=request.proposed_action,
        workflow_thread_id=thread_id,
    )

    if approval is None:
        raise HTTPException(
            status_code=400,
            detail="Could not create approval request.",
        )

    try:
        app.state.approval_graph.invoke(
            {
                "approval_id": approval["id"],
                "proposed_action": approval["proposed_action"],
                "decision": "",
            },
            config={"configurable": {"thread_id": thread_id}},
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Could not start the approval workflow.",
        ) from exc

    return {
        "approval": approval,
        "status": "pending",
        "message": "Approval request created and is waiting for human review.",
    }
