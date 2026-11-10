import json
from pathlib import Path


DATASET_PATH = (
    Path(__file__).parent
    / "datasets"
    / "incident_cases.json"
)


REQUIRED_FIELDS = {
    "case_id",
    "category",
    "difficulty",
    "input",
    "reference_context",
    "expected_behaviour",
    "expected_category",
    "should_escalate"
}


def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def validate_dataset(cases):

    if not isinstance(cases, list) or not cases:
        raise ValueError("Dataset must be a non-empty list.")

    seen_ids = set()

    for case in cases:

        missing = REQUIRED_FIELDS - case.keys()

        if missing:
            raise ValueError(
                f"Missing fields in case: {missing}"
            )

        case_id = case["case_id"]

        if case_id in seen_ids:
            raise ValueError(
                f"Duplicate case ID: {case_id}"
            )

        seen_ids.add(case_id)

        if not isinstance(case["input"], str) or not case["input"].strip():
            raise ValueError(f"Invalid input in {case_id}")

        if not isinstance(case["reference_context"], list):
            raise ValueError(f"Invalid reference_context in {case_id}")

        if not isinstance(case["expected_behaviour"], list):
            raise ValueError(f"Invalid expected_behaviour in {case_id}")

        if not all(
            isinstance(item, str) and item.strip()
            for item in case["expected_behaviour"]
        ):
            raise ValueError(f"Invalid expected_behaviour entries in {case_id}")

        if not isinstance(case["should_escalate"], bool):
            raise ValueError(f"Invalid escalation value in {case_id}")

    return True


if __name__ == "__main__":

    dataset = load_dataset()

    validate_dataset(dataset)

    print("Dataset validation successful!")
    print(f"Total evaluation cases: {len(dataset)}")