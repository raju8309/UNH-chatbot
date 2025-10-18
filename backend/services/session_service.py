from datetime import datetime
from typing import Any, Dict, Optional

# in-memory session store
SESSIONS: Dict[str, Dict[str, Any]] = {}

def _now_iso() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def get_session(session_id: str) -> Dict[str, Any]:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {
            "intent": None,
            "program_level": "unknown",
            "program_alias": None,
            "course_code": None,
            "last_question": None,
            "last_answer": None,
            "last_retrieval_path": None,
            "updated_at": _now_iso(),
        }
    return SESSIONS[session_id]

def update_session(session_id: str, **fields: Any) -> None:
    sess = get_session(session_id)
    sess.update(fields)
    sess["updated_at"] = _now_iso()


def clear_session(session_id: str) -> bool:
    if session_id in SESSIONS:
        del SESSIONS[session_id]
        return True
    return False

def clear_all_sessions() -> None:
    SESSIONS.clear()

def get_all_sessions() -> Dict[str, Dict[str, Any]]:
    return SESSIONS