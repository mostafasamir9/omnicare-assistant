from fastapi import APIRouter, HTTPException
import logging
import traceback

from backend.app.agent.orchestrator import handle_chat, handle_confirm
from backend.app.schemas import (
    ChatRequest,
    ChatResponse,
    ConfirmRequest,
)

logger = logging.getLogger("omnicare.chat")
router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")
    try:
        return handle_chat(req.session_id, req.user_id, req.message)
    except Exception as e:
        logger.error("chat handler failed: %s", e)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@router.post("/confirm", response_model=ChatResponse)
def confirm(req: ConfirmRequest):
    try:
        return handle_confirm(
            req.session_id, req.user_id, req.action_id, req.confirm
        )
    except Exception as e:
        logger.error("confirm handler failed: %s", e)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")