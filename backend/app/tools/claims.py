import re

from backend.data import mock_db
from backend.app.schemas import ClaimStatusResult, ClaimSubmissionResult

CLAIM_ID_RE = re.compile(r"\bCLM-\d{4}\b", re.IGNORECASE)


def extract_claim_id(text: str) -> str | None:
    m = CLAIM_ID_RE.search(text)
    return m.group(0).upper() if m else None


def get_claim_status(claim_id: str, user_id: str) -> ClaimStatusResult | None:
    data = mock_db.get_claim_status(claim_id)
    if not data:
        return None
    return ClaimStatusResult(**data)


def submit_claim(
    policy_id: str,
    user_id: str,
    claim_type: str,
    incident_date: str,
    description: str,
    amount_estimate: float,
) -> ClaimSubmissionResult:
    if not mock_db.policy_belongs_to_user(policy_id, user_id):
        return ClaimSubmissionResult(
            claim_id="",
            status="rejected",
            message=f"Policy {policy_id} not found or not owned by this user.",
        )
    result = mock_db.create_claim(
        policy_id, claim_type, incident_date, description, amount_estimate
    )
    return ClaimSubmissionResult(
        claim_id=result["claim_id"],
        status=result["status"],
        message=f"Claim {result['claim_id']} submitted successfully.",
    )