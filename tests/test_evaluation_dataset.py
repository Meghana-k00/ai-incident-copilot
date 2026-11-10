from evaluation.validate_dataset import (
    load_dataset,
    validate_dataset
)


def test_dataset_is_valid():
    cases = load_dataset()
    assert validate_dataset(cases) is True


def test_dataset_has_unique_ids():
    cases = load_dataset()

    ids = [case["case_id"] for case in cases]

    assert len(ids) == len(set(ids))


def test_dataset_contains_expected_fields():
    cases = load_dataset()

    for case in cases:
        assert "input" in case
        assert "reference_context" in case
        assert "expected_behaviour" in case