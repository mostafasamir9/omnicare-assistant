"""Mock operational systems: claims DB and claims intake."""
import uuid
from datetime import datetime

_CLAIMS_DB: dict[str, dict] = {
    "CLM-1001": {
        "claim_id": "CLM-1001",
        "status": "Under Review",
        "filed_date": "2024-08-12",
        "last_update": "2024-09-01",
        "adjuster": "J. Rivera",
        "notes": "Awaiting repair estimate from body shop.",
    },
    "CLM-1002": {
        "claim_id": "CLM-1002",
        "status": "Approved",
        "filed_date": "2024-07-03",
        "last_update": "2024-08-15",
        "adjuster": "M. Chen",
        "notes": "Payment issued. Check mailed 2024-08-15.",
    },
    "CLM-1003": {
        "claim_id": "CLM-1003",
        "status": "Denied",
        "filed_date": "2024-06-20",
        "last_update": "2024-07-10",
        "adjuster": "S. Patel",
        "notes": "Damage occurred during racing event — excluded per Section 2.1.",
    },
}

_POLICIES: dict[str, dict] = {
    "POL-001": {"user_id": "user-1", "type": "auto", "state": "CA"},
    "POL-002": {"user_id": "user-1", "type": "home", "state": "CA"},
    "POL-003": {"user_id": "user-2", "type": "life", "state": "CA"},
}


def get_claim_status(claim_id: str) -> dict | None:
    return _CLAIMS_DB.get(claim_id)


def policy_belongs_to_user(policy_id: str, user_id: str) -> bool:
    p = _POLICIES.get(policy_id)
    return bool(p and p["user_id"] == user_id)


def create_claim(
    policy_id: str,
    claim_type: str,
    incident_date: str,
    description: str,
    amount_estimate: float,
) -> dict:
    new_id = f"CLM-{uuid.uuid4().hex[:6].upper()}"
    record = {
        "claim_id": new_id,
        "status": "Submitted",
        "filed_date": datetime.utcnow().strftime("%Y-%m-%d"),
        "last_update": datetime.utcnow().strftime("%Y-%m-%d"),
        "adjuster": "Unassigned",
        "notes": f"New {claim_type} claim on {policy_id}. Est: ${amount_estimate:.2f}. {description[:120]}",
    }
    _CLAIMS_DB[new_id] = record
    return record