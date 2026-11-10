import json
import pytest

from unittest.mock import MagicMock, patch

from evaluation.faithfulness import (
    evaluate_faithfulness,
    FaithfulnessEvaluationError,
    _calculate_score,
)


def mock_ollama_response(judgement):
    response = MagicMock()
    response.__enter__.return_value.read.return_value = (
        json.dumps({
            "response": json.dumps(judgement)
        }).encode("utf-8")
    )
    return response


def test_python_calculates_score_from_claims():
    claims = [
        {"verdict": "supported"},
        {"verdict": "supported"},
        {"verdict": "unsupported"},
        {"verdict": "not_verifiable"},
        {"verdict": "hypothesis"},
        {"verdict": "recommendation"},
    ]

    result = _calculate_score(claims)

    assert result["score"] == 0.5
    assert result["factual_claim_count"] == 4
    assert result["supported_count"] == 2
    assert result["unsupported_count"] == 1
    assert result["not_verifiable_count"] == 1
    assert result["hypothesis_count"] == 1
    assert result["recommendation_count"] == 1


def test_evaluator_recognises_supported_claim():
    judgement = {
        "claims": [
            {
                "claim": "The application reports database timeouts.",
                "verdict": "supported",
                "evidence_ids": ["1"],
                "reason": "The incident report explicitly states this.",
            }
        ],
        "reasoning": "The claim is supported by the incident report.",
    }

    with patch(
        "evaluation.faithfulness.urllib.request.urlopen",
        return_value=mock_ollama_response(judgement),
    ):
        result = evaluate_faithfulness(
            response="The application reports database timeouts.",
            evidence=["The application reports database connection timeouts."],
        )

    assert result["status"] == "pass"
    assert result["score"] == 1.0
    assert result["counts"]["supported_count"] == 1


def test_hypotheses_and_recommendations_are_not_in_score():
    judgement = {
        "claims": [
            {
                "claim": "Connection pool exhaustion may be responsible.",
                "verdict": "hypothesis",
                "evidence_ids": [],
                "reason": "It is explicitly labelled as a possibility.",
            },
            {
                "claim": "Check the database connection pool.",
                "verdict": "recommendation",
                "evidence_ids": [],
                "reason": "This is a proposed diagnostic action.",
            },
        ],
        "reasoning": "There are no factual claims to score.",
    }

    with patch(
        "evaluation.faithfulness.urllib.request.urlopen",
        return_value=mock_ollama_response(judgement),
    ):
        result = evaluate_faithfulness(
            response="Check the database connection pool.",
            evidence=["The database connection timed out."],
        )

    assert result["status"] == "no_factual_claims"
    assert result["score"] is None
    assert result["counts"]["hypothesis_count"] == 1
    assert result["counts"]["recommendation_count"] == 1


def test_unsupported_claim_fails_evaluation():
    judgement = {
        "claims": [
            {
                "claim": "The database server has crashed.",
                "verdict": "unsupported",
                "evidence_ids": [],
                "reason": "The evidence only mentions connection timeouts.",
            }
        ],
        "reasoning": "The evidence does not establish a server crash.",
    }

    with patch(
        "evaluation.faithfulness.urllib.request.urlopen",
        return_value=mock_ollama_response(judgement),
    ):
        result = evaluate_faithfulness(
            response="The database server has crashed.",
            evidence=["The database connection timed out."],
        )

    assert result["status"] == "fail"
    assert result["score"] == 0.0
    assert result["counts"]["unsupported_count"] == 1


def test_missing_evidence_returns_insufficient_status():
    result = evaluate_faithfulness(
        response="The database server is unavailable.",
        evidence=[],
    )

    assert result["status"] == "insufficient_evidence"
    assert result["score"] is None


def test_evaluator_rejects_unknown_verdict():
    judgement = {
        "claims": [
            {
                "claim": "The database is unavailable.",
                "verdict": "probably_true",
                "evidence_ids": ["1"],
                "reason": "Invalid classification.",
            }
        ],
        "reasoning": "Invalid verdict test.",
    }

    with patch(
        "evaluation.faithfulness.urllib.request.urlopen",
        return_value=mock_ollama_response(judgement),
    ):
        with pytest.raises(FaithfulnessEvaluationError):
            evaluate_faithfulness(
                response="The database is unavailable.",
                evidence=["The database connection timed out."],
            )


def test_evaluator_rejects_unknown_evidence_id():
    judgement = {
        "claims": [
            {
                "claim": "The database is unavailable.",
                "verdict": "supported",
                "evidence_ids": ["99"],
                "reason": "Invalid evidence reference.",
            }
        ],
        "reasoning": "Invalid evidence ID test.",
    }

    with patch(
        "evaluation.faithfulness.urllib.request.urlopen",
        return_value=mock_ollama_response(judgement),
    ):
        with pytest.raises(FaithfulnessEvaluationError):
            evaluate_faithfulness(
                response="The database is unavailable.",
                evidence=["The database connection timed out."],
            )


def test_evaluator_rejects_invalid_json():
    response = MagicMock()
    response.__enter__.return_value.read.return_value = (
        b'{"response": "not valid JSON"}'
    )

    with patch(
        "evaluation.faithfulness.urllib.request.urlopen",
        return_value=response,
    ):
        with pytest.raises(FaithfulnessEvaluationError):
            evaluate_faithfulness(
                response="The database is unavailable.",
                evidence=["The database connection timed out."],
            )


def test_invalid_threshold_is_rejected():
    with pytest.raises(ValueError):
        evaluate_faithfulness(
            response="The database is unavailable.",
            evidence=["The database connection timed out."],
            threshold=1.5,
        )