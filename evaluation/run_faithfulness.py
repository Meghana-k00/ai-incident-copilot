import json

from src.investigation import investigate_incident
from src.retrieval import IncidentRetriever
from evaluation.faithfulness import evaluate_faithfulness


def build_incident():
    """Create the incident used for this evaluation run."""
    return {
        "id": 1,
        "title": "Database connection timeout",
        "description": (
            "The application cannot connect to the database. "
            "Requests are failing because of connection timeouts."
        ),
        "severity": "high",
        "service_name": "customer-api",
        "status": "open",
    }


def main():
    retriever = IncidentRetriever()
    incident = build_incident()

    print("\nRunning AI incident investigation...")

    investigation = investigate_incident(
        incident,
        retriever=retriever,
    )

    print("\nGenerated investigation:")
    print(investigation["investigation"])

    # Evidence from the current incident report.
    current_incident_evidence = [
        {
            "text": (
                f"Current incident report:\n"
                f"Title: {incident['title']}\n"
                f"Description: {incident['description']}\n"
                f"Severity: {incident['severity']}\n"
                f"Service: {incident['service_name']}\n"
                f"Status: {incident['status']}"
            )
        }
    ]

    # Historical documents are reference material, not
    # verified facts about the current incident.
    historical_evidence = [
        {
            "text": (
                "Historical reference (not proof of the current incident):\n"
                + document["text"]
            )
        }
        for document in investigation["retrieved_documents"]
    ]

    all_evidence = (
        current_incident_evidence + historical_evidence
    )

    print("\nRunning faithfulness evaluation...")

    evaluation = evaluate_faithfulness(
        response=investigation["investigation"],
        evidence=all_evidence,
    )

    print("\nFaithfulness evaluation result:")
    print(json.dumps(evaluation, indent=4))


if __name__ == "__main__":
    main()