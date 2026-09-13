from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class Citation(BaseModel):
    doc_id: str
    section: str
    page: int
    snippet: str


class ToolCallRecord(BaseModel):
    name: str
    args: dict[str, Any]
    result: Any = None
    ok: bool = True
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ChatRequest(BaseModel):
    session_id: str
    user_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    citations: list[Citation] = []
    tool_calls: list[ToolCallRecord] = []
    pending_action: Optional[dict] = None
    refused: bool = False


class ConfirmRequest(BaseModel):
    session_id: str
    user_id: str
    action_id: str
    confirm: bool


class ClaimStatusResult(BaseModel):
    claim_id: str
    status: str
    filed_date: str
    last_update: str
    adjuster: str
    notes: str


class ClaimSubmission(BaseModel):
    policy_id: str
    claim_type: Literal["auto", "home", "life", "health"]
    incident_date: str
    description: str
    amount_estimate: float


class ClaimSubmissionResult(BaseModel):
    claim_id: str
    status: str
    message: str