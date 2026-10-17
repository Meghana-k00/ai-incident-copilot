import json
import urllib.error
import urllib.request

from typing import Any


def prepare_investigation(incident: dict[str, Any]) -> dict[str, Any]:
    """
    Prepare structured context for investigating an incident.
    This function does not call an AI model or modify the database.
    """
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


def investigate_incident(
    incident: dict[str, Any],
    model: str = "qwen2.5:3b",
) -> dict[str, Any]:
    """Generate AI investigation findings using local Ollama."""

    context = prepare_investigation(incident)

    prompt = f"""
You are an experienced site reliability engineer.
Analyze the incident below.

Incident:
{json.dumps(context, indent=2)}

Provide:
1. Three plausible root causes.
2. Diagnostic checks to investigate them.
3. Recommended remediation steps.
4. A short explanation of your reasoning.

Do not claim that you have verified a cause.
Return your findings as plain text.
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

    return {
        "incident_id": context["incident_id"],
        "model": model,
        "investigation": answer.strip(),
    }