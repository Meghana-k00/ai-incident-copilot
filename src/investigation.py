
import json
import urllib.error
import urllib.request

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from src.retrieval import IncidentRetriever


class InvestigationState(TypedDict, total=False):
    incident: dict[str, Any]
    model: str
    retriever: IncidentRetriever | None
    context: dict[str, Any]
    retrieved_documents: list[dict[str, Any]]
    investigation: str
    result: dict[str, Any]


def prepare_investigation(incident: dict[str, Any]) -> dict[str, Any]:
    """Validate an incident and prepare structured investigation context."""
    title = str(incident.get("title", "")).strip()
    description = str(incident.get("description", "")).strip()
    severity = str(incident.get("severity", "unknown")).strip().lower()
    service_name = str(incident.get("service_name", "")).strip()
    status = str(incident.get("status", "open")).strip().lower()

    if not title or not description or not service_name:
        raise ValueError(
            "Incident title, description, and service name are required."
        )

    if severity not in {"low", "medium", "high", "critical"}:
        raise ValueError("Unsupported incident severity.")

    return {
        "incident_id": incident.get("id"),
        "title": title,
        "description": description,
        "severity": severity,
        "service_name": service_name,
        "status": status,
        "investigation_context": (
            f"Investigate the incident '{title}' affecting "
            f"'{service_name}'. Severity: {severity}. "
            f"Current status: {status}. "
            f"Reported details: {description}"
        ),
    }


def prepare_incident_node(state: InvestigationState) -> dict[str, Any]:
    """Graph node 1: validate and prepare the incident."""
    return {
        "context": prepare_investigation(state["incident"])
    }


def retrieve_knowledge_node(state: InvestigationState) -> dict[str, Any]:
    """Graph node 2: retrieve relevant historical incident knowledge."""
    retriever = state.get("retriever")
    documents: list[dict[str, Any]] = []

    if retriever is not None:
        documents = retriever.search(
            state["context"]["investigation_context"],
            top_k=3,
        )

    return {"retrieved_documents": documents}


def generate_investigation_node(
    state: InvestigationState,
) -> dict[str, Any]:
    """Graph node 3: generate investigation findings using local Ollama."""
    context = state["context"]
    documents = state.get("retrieved_documents", [])
    model = state.get("model", "qwen2.5:3b")

    historical_knowledge = (
        "\n\n".join(
            f"Reference {index}:\n{document['text']}"
            for index, document in enumerate(documents, start=1)
        )
        or "No relevant historical incident knowledge was retrieved."
    )

    prompt = f"""
You are an experienced site reliability engineer.
Analyze the incident below.

Current incident:
{json.dumps(context, indent=2)}

Relevant historical knowledge retrieved from the knowledge base:
{historical_knowledge}

Instructions:
1. Provide three plausible root causes.
2. Recommend diagnostic checks for each likely cause.
3. Suggest safe, actionable remediation steps.
4. Explain your reasoning.
5. Use historical knowledge only when relevant to the current incident.
6. Distinguish historical information from facts about the current incident.
7. Do not claim that a cause has been verified.
8. Return your findings as plain text.
"""

    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
    }).encode("utf-8")

    request = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))

    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"Ollama returned HTTP error {exc.code}."
        ) from exc

    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Could not connect to Ollama. Make sure Ollama is running."
        ) from exc

    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError(
            "Ollama returned an invalid response."
        ) from exc

    if not isinstance(result, dict):
        raise RuntimeError(
            "Ollama returned an unexpected response format."
        )

    answer = result.get("response")

    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError(
            "Ollama returned an empty or invalid investigation."
        )

    return {"investigation": answer.strip()}


def finalize_investigation_node(
    state: InvestigationState,
) -> dict[str, Any]:
    """Graph node 4: assemble the existing public response format."""
    context = state["context"]

    return {
        "result": {
            "incident_id": context["incident_id"],
            "model": state.get("model", "qwen2.5:3b"),
            "investigation": state["investigation"],
            "retrieved_documents": state.get("retrieved_documents", []),
        }
    }


def build_investigation_graph():
    """Build and compile the incident investigation workflow."""
    workflow = StateGraph(InvestigationState)

    workflow.add_node("prepare_incident", prepare_incident_node)
    workflow.add_node("retrieve_knowledge", retrieve_knowledge_node)
    workflow.add_node("generate_investigation", generate_investigation_node)
    workflow.add_node("finalize_investigation", finalize_investigation_node)

    workflow.add_edge(START, "prepare_incident")
    workflow.add_edge("prepare_incident", "retrieve_knowledge")
    workflow.add_edge("retrieve_knowledge", "generate_investigation")
    workflow.add_edge("generate_investigation", "finalize_investigation")
    workflow.add_edge("finalize_investigation", END)

    return workflow.compile()


investigation_graph = build_investigation_graph()


def investigate_incident(
    incident: dict[str, Any],
    model: str = "qwen2.5:3b",
    retriever: IncidentRetriever | None = None,
) -> dict[str, Any]:
    """Run incident investigation through the compiled LangGraph workflow."""
    state: InvestigationState = {
        "incident": incident,
        "model": model,
        "retriever": retriever,
    }

    result = investigation_graph.invoke(state)
    return result["result"]
