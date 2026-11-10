import json
import urllib.error
import urllib.request

from typing import Any


DEFAULT_MODEL = "qwen2.5:3b"
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_THRESHOLD = 0.80

ALLOWED_VERDICTS = {
    "supported",
    "unsupported",
    "not_verifiable",
    "hypothesis",
    "recommendation",
}


class FaithfulnessEvaluationError(RuntimeError):
    """Raised when faithfulness evaluation cannot be completed."""


def _normalise_evidence(
    evidence: list[str] | list[dict[str, Any]],
) -> list[str]:
    """Convert retrieved documents or strings into evidence text."""
    if not isinstance(evidence, list):
        raise ValueError("Evidence must be a list.")

    normalised = []

    for item in evidence:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            text = str(item.get("text", "")).strip()
        else:
            raise ValueError(
                "Evidence entries must be strings or dictionaries."
            )

        if text:
            normalised.append(text)

    return normalised


def _validate_judgement(
    result: Any,
    evidence_count: int,
) -> dict[str, Any]:
    """Validate individual claim judgements from the LLM."""
    if not isinstance(result, dict):
        raise FaithfulnessEvaluationError(
            "Evaluator must return a JSON object."
        )

    claims = result.get("claims")
    reasoning = result.get("reasoning")

    if not isinstance(claims, list):
        raise FaithfulnessEvaluationError(
            "Evaluator must return a claims list."
        )

    if not isinstance(reasoning, str) or not reasoning.strip():
        raise FaithfulnessEvaluationError(
            "Evaluator must provide overall reasoning."
        )

    validated_claims = []

    for index, claim in enumerate(claims):
        if not isinstance(claim, dict):
            raise FaithfulnessEvaluationError(
                f"Claim {index + 1} must be an object."
            )

        statement = claim.get("claim")
        raw_verdict = claim.get("verdict")
        evidence_ids = claim.get("evidence_ids")
        reason = claim.get("reason") or claim.get("explanation") or claim.get("reasoning")

        if not isinstance(statement, str) or not statement.strip():
            raise FaithfulnessEvaluationError(
                f"Claim {index + 1} has invalid text."
            )

        if not isinstance(raw_verdict, str):
            raise FaithfulnessEvaluationError(
                f"Claim {index + 1} has a non-string verdict: {raw_verdict!r}"
            )

        verdict = raw_verdict.strip().lower()

        if verdict not in ALLOWED_VERDICTS:
            raise FaithfulnessEvaluationError(
                f"Claim {index + 1} has an invalid verdict: {raw_verdict!r}. "
                f"Allowed values: {sorted(ALLOWED_VERDICTS)}"
            )
        if not isinstance(evidence_ids, list) or not all(
            isinstance(item, str) for item in evidence_ids
        ):
            raise FaithfulnessEvaluationError(
                f"Claim {index + 1} has invalid evidence IDs."
            )

        valid_ids = {
            str(number)
            for number in range(1, evidence_count + 1)
        }

        if not set(evidence_ids).issubset(valid_ids):
            raise FaithfulnessEvaluationError(
                f"Claim {index + 1} references unknown evidence."
            )

        if verdict == "supported" and not evidence_ids:
            raise FaithfulnessEvaluationError(
                f"Supported claim {index + 1} must reference evidence."
            )
            
        if not isinstance(reason, str) or not reason.strip():
            raise FaithfulnessEvaluationError(
                f"Claim {index + 1} has no explanation."
            )

        validated_claims.append({
            "claim": statement.strip(),
            "verdict": verdict,
            "evidence_ids": evidence_ids,
            "reason": reason.strip(),
        })

    return {
        "claims": validated_claims,
        "reasoning": reasoning.strip(),
    }


def _calculate_score(claims: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate faithfulness deterministically in Python."""
    factual_verdicts = {
        "supported",
        "unsupported",
        "not_verifiable",
    }

    factual_claims = [
        claim
        for claim in claims
        if claim["verdict"] in factual_verdicts
    ]

    supported_count = sum(
        claim["verdict"] == "supported"
        for claim in factual_claims
    )

    unsupported_count = sum(
        claim["verdict"] == "unsupported"
        for claim in factual_claims
    )

    not_verifiable_count = sum(
        claim["verdict"] == "not_verifiable"
        for claim in factual_claims
    )

    if not factual_claims:
        score = None
    else:
        score = round(
            supported_count / len(factual_claims),
            4,
        )

    return {
        "score": score,
        "factual_claim_count": len(factual_claims),
        "supported_count": supported_count,
        "unsupported_count": unsupported_count,
        "not_verifiable_count": not_verifiable_count,
        "hypothesis_count": sum(
            claim["verdict"] == "hypothesis"
            for claim in claims
        ),
        "recommendation_count": sum(
            claim["verdict"] == "recommendation"
            for claim in claims
        ),
    }


def evaluate_faithfulness(
    response: str,
    evidence: list[str] | list[dict[str, Any]],
    model: str = DEFAULT_MODEL,
    threshold: float = DEFAULT_THRESHOLD,
    timeout: int = 120,
) -> dict[str, Any]:
    """
    Evaluate response claims against evidence using a local Ollama judge.

    The LLM classifies claims. Python calculates the final score.
    """

    if not isinstance(response, str) or not response.strip():
        raise ValueError("Response must be a non-empty string.")

    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or not 0 <= threshold <= 1
    ):
        raise ValueError("Threshold must be between 0 and 1.")

    evidence_items = _normalise_evidence(evidence)

    if not evidence_items:
        return {
            "status": "insufficient_evidence",
            "score": None,
            "threshold": threshold,
            "claims": [],
            "counts": {},
            "reasoning": (
                "No evidence was provided. "
                "Faithfulness cannot be assessed."
            ),
            "model": model,
        }

    evidence_text = "\n\n".join(
        f"Evidence {index}:\n{item}"
        for index, item in enumerate(evidence_items, start=1)
    )

    prompt = f"""
You are a strict claim-level factual grounding evaluator.

EVIDENCE:
{evidence_text}

AI-GENERATED RESPONSE:
{response}

Evaluate each meaningful statement individually.

Allowed classifications:

SUPPORTED:
Directly supported by the supplied evidence. Accurate paraphrases
are allowed.

UNSUPPORTED:
Asserted as a fact, but not established by the evidence.

NOT_VERIFIABLE:
There is insufficient evidence to confirm or reject the claim.

HYPOTHESIS:
A potential cause explicitly labelled as possible or unverified.

RECOMMENDATION:
A proposed diagnostic check or action, not a claim that it happened.

Rules:
1. The current incident report is evidence of what was reported.
2. Historical references are not proof of the current incident.
3. A reported timeout is supported if the incident report states it.
4. Recommendations are not factual claims.
5. Clearly labelled hypotheses are not confirmed facts.
6. Do not classify an asserted but unverified cause as a hypothesis.
7. Every supported claim must identify at least one evidence ID.
8. Use only the supplied evidence IDs.
9. Do not invent evidence or claim that checks were performed.
10. Evaluate every meaningful factual statement.

MANDATORY CLAIM EXTRACTION PROCEDURE:

A. Read the complete AI-GENERATED RESPONSE before assigning verdicts.

B. Examine each section separately:
   - Evidence-supported observations
   - Unverified root-cause hypotheses
   - Recommended diagnostic checks
   - Safe remediation suggestions
   - Evidence gaps

C. Extract EVERY factual observation from the
   "Evidence-supported observations" section.
   Each observation bullet must produce a separate claim.

D. Do not omit a factual observation because another
   section contains hypotheses about the same incident.

E. Classify possible causes in the hypotheses section
   as hypotheses, not factual claims.

F. Classify diagnostic checks and proposed actions
   as recommendations.

G. Before returning JSON, verify that every factual
   observation bullet has a corresponding claim.

H. Never invent a factual observation to meet this requirement.

Return ONLY valid JSON matching this structure:

{{
  "claims": [
    {{
      "claim": "Exact factual statement extracted from the response",
      "verdict": "supported",
      "evidence_ids": ["1"],
      "reason": "Explanation based on the supplied evidence"
    }}
  ],
  "reasoning": "Brief explanation of the overall evaluation."
}}

STRICT EVALUATION RULES:

1. Extract claims only from the AI-GENERATED RESPONSE.
2. Do not copy claims from this instruction or create example claims.
3. Do not introduce database, network, authentication, or other
   technical claims unless they actually appear in the response.
4. Preserve the meaning of each extracted statement.
5. Evaluate every meaningful factual statement.
6. Classify a claim as supported only when the evidence directly
   establishes it.
7. Classify an asserted but unsubstantiated fact as unsupported.
8. Use not_verifiable when the available evidence cannot establish
   whether the claim is true or false.
9. Use hypothesis only for a potential cause explicitly labelled
   as unverified or possible in the response.
10. Use recommendation for proposed checks or actions.
11. Recommendations and hypotheses must not be counted as factual claims.
12. Every supported claim must reference valid evidence IDs.
13. Do not treat historical references as proof of the current incident.
14. Do not treat diagnostic recommendations as completed actions.
15. If the response contains no meaningful claims, return an empty list.
16. Return valid JSON only. Do not include Markdown fences.

STRICT EVIDENCE MATCHING RULES:

0. VERDICT CLASSIFICATION:
   - If a claim is explicitly labelled UNVERIFIED, or clearly
     presented as a possible cause using words such as may,
     might, or could, classify it as "hypothesis".
   - Do not classify a clearly labelled hypothesis as
     "unsupported" merely because evidence does not confirm it.
   - If a statement is presented as an established fact but
     lacks supporting evidence, classify it as "unsupported".
   - Diagnostic checks and remediation suggestions must be
     classified as "recommendation", not factual claims.
1. Do not add details that are absent from the evidence.
2. An incident description is not automatically a log, monitoring
   result, or diagnostic record.
3. Never change the source of information. If the incident report
   mentions a timeout, do not claim that logs show a timeout.
4. A claim is supported only when the supplied evidence directly
   supports the complete factual statement.
5. If a claim contains both supported and unsupported details,
   classify the complete claim as unsupported.
6. Do not treat possible root causes as confirmed facts.
7. Use not_verifiable when the available evidence cannot establish
   whether a claim is true or false.
"""

    
    current_prompt = prompt
    judgement = None

    for attempt in range(2):
        payload = json.dumps({
            "model": model,
            "prompt": current_prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0,
                "seed": 42,
            },
        }).encode("utf-8")

        request = urllib.request.Request(
            OLLAMA_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=timeout,
            ) as http_response:
                ollama_result = json.loads(
                    http_response.read().decode("utf-8")
                )

        except urllib.error.HTTPError as exc:
            raise FaithfulnessEvaluationError(
                f"Ollama returned HTTP error {exc.code}."
            ) from exc

        except urllib.error.URLError as exc:
            raise FaithfulnessEvaluationError(
                "Cannot connect to Ollama. Check that Ollama is running."
            ) from exc

        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise FaithfulnessEvaluationError(
                "Ollama returned an invalid response."
            ) from exc

        if not isinstance(ollama_result, dict):
            raise FaithfulnessEvaluationError(
                "Ollama returned an unexpected response."
            )

        raw_judgement = ollama_result.get("response")

        if not isinstance(raw_judgement, str) or not raw_judgement.strip():
            raise FaithfulnessEvaluationError(
                "Ollama returned an empty evaluation."
            )

        try:
            parsed_judgement = json.loads(raw_judgement)
        except json.JSONDecodeError as exc:
            raise FaithfulnessEvaluationError(
                "The evaluator did not return valid JSON."
            ) from exc

        try:
            judgement = _validate_judgement(
                parsed_judgement,
                evidence_count=len(evidence_items),
            )
            break

        except FaithfulnessEvaluationError as exc:
            if attempt == 1:
                raise

            current_prompt = f"""
{prompt}

CORRECTION REQUIRED:

Your previous JSON response failed validation:
{str(exc)}

Previous response:
{raw_judgement}

Return a corrected JSON object using the required schema.

Important:
- Every claim must have a non-empty "claim" string.
- Every claim must have a valid "verdict".
- Every claim must have an "evidence_ids" list containing
  only IDs supplied in the EVIDENCE section.
- Every claim must have a non-empty "reason" explaining
  its classification using only the supplied evidence.
- Do not invent evidence or change a claim into a supported
  claim without direct evidence.
- Preserve hypotheses and recommendations as their own
  classifications.
- Do not omit meaningful factual observations.
- Return JSON only.
"""

    if judgement is None:
        raise FaithfulnessEvaluationError(
            "Evaluator could not produce a valid judgement."
        )

    counts = _calculate_score(judgement["claims"])
    score = counts["score"]

    if score is None:
        status = "no_factual_claims"
    elif score >= threshold:
        status = "pass"
    else:
        status = "fail"

    return {
        "status": status,
        "score": score,
        "threshold": threshold,
        "claims": judgement["claims"],
        "counts": counts,
        "reasoning": judgement["reasoning"],
        "model": model,
    }