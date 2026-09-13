from backend.app.tools.claims import extract_claim_id

SUBMIT_KEYWORDS = [
    "file a claim", "file claim", "submit a claim", "submit claim",
    "new claim", "start a claim", "report a claim", "make a claim",
]

STATUS_KEYWORDS = [
    "status", "where is my claim", "check my claim", "claim update",
    "what's happening with my claim", "whats happening with my claim",
]


def deterministic_route(message: str) -> str | None:
    """Return 'status' | 'submit' | None."""
    lowered = message.lower()

    if extract_claim_id(message) and "claim" in lowered:
        return "status"

    if any(k in lowered for k in SUBMIT_KEYWORDS):
        return "submit"

    if any(k in lowered for k in STATUS_KEYWORDS):
        return "status"

    return None