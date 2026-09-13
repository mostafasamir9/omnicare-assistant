import time
import uuid
from collections import defaultdict

from backend.app.config import get_settings

_settings = get_settings()


class SessionState:
    """In-memory session store. Swap for Redis in production."""

    def __init__(self) -> None:
        self._store: dict[str, dict] = defaultdict(self._new_session)
        self._pending_actions: dict[str, dict] = {}

    def _new_session(self) -> dict:
        return {
            "history": [],
            "tokens_used": 0,
            "tool_calls": 0,
            "created_at": time.time(),
        }

    def get(self, session_id: str) -> dict:
        return self._store[session_id]

    def history(self, session_id: str) -> list[dict]:
        return self._store[session_id]["history"]

    def append(self, session_id: str, role: str, content: str) -> None:
        self._store[session_id]["history"].append({"role": role, "content": content})

    def add_tokens(self, session_id: str, n: int) -> bool:
        s = self._store[session_id]
        s["tokens_used"] += n
        return s["tokens_used"] <= _settings.max_tokens_per_session

    def add_tool_call(self, session_id: str) -> bool:
        s = self._store[session_id]
        s["tool_calls"] += 1
        return s["tool_calls"] <= _settings.max_tool_calls_per_session

    def stash_pending(self, session_id: str, action: dict) -> str:
        action_id = str(uuid.uuid4())
        action["action_id"] = action_id
        self._pending_actions[action_id] = {"session_id": session_id, **action}
        return action_id

    def pop_pending(self, action_id: str) -> dict | None:
        return self._pending_actions.pop(action_id, None)

    def get_pending(self, action_id: str) -> dict | None:
        return self._pending_actions.get(action_id)


session_store = SessionState()