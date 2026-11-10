import argparse
import json

from datetime import datetime, timezone
from pathlib import Path

from src.investigation import investigate_incident
from src.retrieval import IncidentRetriever
from evaluation.faithfulness import evaluate_faithfulness


EVALUATION_DIR = Path(__file__).parent
DATASET_PATH = EVALUATION_DIR / "datasets" / "incident_cases.json"
REPORT_PATH = EVALUATION_DIR / "reports" / "faithfulness_report.json"

SKIP_CATEGORIES = {
    "out_of_scope",
    "approval_required",
    "prompt_injection",
}
SKIP_REASONS = {
    "out_of_scope": "Requires separate out-of-scope behaviour evaluation.",
    "approval_required": "Requires separate human-approval workflow evaluation.",
    "prompt_injection": "Requires separate prompt-injection robustness evaluation.",
}
def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def build_incident(case):
    """Convert a golden case into the existing incident input format."""

    severity = {
        "easy": "low",
        "medium": "medium",
        "hard": "high",
    }.get(case["difficulty"], "medium")

    return {
        "id": case["case_id"],
        "title": f"{case['category'].replace('_', ' ').title()} incident",
        "description": case["input"],
        "severity": severity,
        "service_name": "evaluation-service",
        "status": "open",
        "metadata_is_synthetic": True,
    }



def evaluate_case(case, retriever):
    """Run one golden case through investigation and faithfulness evaluation."""

    # 1. Build the incident from the golden dataset.
    incident = build_incident(case)

    # 2. Run the investigation pipeline.
    investigation = investigate_incident(
        incident,
        retriever=retriever,
    )
    

    # 3. Start evidence with the current incident report.
    evidence = [
        {
            "text": f"Incident report: {case['input']}"
        }
    ]

    # 4. Add the reference evidence for this test case.
    evidence.extend(
        {
            "text": f"Reference evidence: {item}"
        }
        for item in case["reference_context"]
    )

    # 5. Add retrieved historical documents as separate evidence.
    # Historical documents must not be treated as proof of the
    # current incident.
    evidence.extend(
        {
            "text": (
                "Historical reference, not proof of the current incident: "
                + document["text"]
            )
        }
        for document in investigation["retrieved_documents"]
    )

    # 8. Evaluate the investigation's faithfulness.
    result = evaluate_faithfulness(
        response=investigation["investigation"],
        evidence=evidence,
    )

    # 9. Return a structured evaluation result.
    return {
        "case_id": case["case_id"],
        "category": case["category"],
        "status": result["status"],
        "score": result["score"],
        "counts": result["counts"],
        "claims": result["claims"],
        "reasoning": result["reasoning"],
        "investigation": investigation["investigation"],
    }



def main():
    parser = argparse.ArgumentParser(
        description="Run the AI Incident Copilot golden dataset evaluation."
    )
    parser.add_argument(
        "--case-id",
        help="Run one case only, for example INC001.",
    )
    args = parser.parse_args()

    cases = load_dataset()

    selected_cases = [
        case for case in cases
        if case["category"] not in SKIP_CATEGORIES
        and case["reference_context"]
    ]

    skipped_cases = [
        {
            "case_id": case["case_id"],
            "category": case["category"],
            "reason": SKIP_REASONS.get(
                case["category"],
                "No reference evidence available for faithfulness evaluation."
            ),
        }
        for case in cases
        if case not in selected_cases
    ]

    if args.case_id:
        selected_cases = [
            case for case in selected_cases
            if case["case_id"] == args.case_id
        ]

        if not selected_cases:
            raise ValueError(
                f"No eligible golden case found: {args.case_id}"
            )

    retriever = IncidentRetriever()

    results = []

    for case in selected_cases:
        print(f"\nEvaluating {case['case_id']}...")

        try:
            result = evaluate_case(case, retriever)
            results.append(result)

            print(
                f"Status: {result['status']} | "
                f"Score: {result['score']}"
            )

        except Exception as exc:
            results.append({
                "case_id": case["case_id"],
                "status": "error",
                "error": str(exc),
            })

            print(f"Evaluation error: {exc}")

    status_counts = {}

    for item in results:
        status = item["status"]
        status_counts[status] = status_counts.get(status, 0) + 1

    completed = len(results) - status_counts.get("error", 0)

    passed = status_counts.get("pass", 0)
    failed = status_counts.get("fail", 0)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": "qwen2.5:3b",
        "total_dataset_cases": len(cases),
        "cases_selected": len(selected_cases),
        "cases_skipped": skipped_cases,
        "summary": {
        "completed": completed,
        "passed": passed,
        "failed": failed,
        "errors": status_counts.get("error", 0),
        "status_counts": status_counts,
    },
        "results": results,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\n========== EVALUATION SUMMARY ==========")
    print(f"Dataset cases: {len(cases)}")
    print(f"Evaluated: {len(selected_cases)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Errors: {report['summary']['errors']}")
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()