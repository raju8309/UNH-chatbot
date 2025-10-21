from datetime import datetime
from typing import Any, Dict, Optional

# in-memory session store
SESSIONS: Dict[str, Dict[str, Any]] = {}

def now_iso() -> str:
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
            "history": [],
            "updated_at": now_iso(),
        }
    return SESSIONS[session_id]

def update_session(session_id: str, **fields: Any) -> None:
    sess = get_session(session_id)
    sess.update(fields)
    sess["updated_at"] = now_iso()

# -----------------------
# Session history helper
# -----------------------
def push_history(session_id: str, entry: Dict[str, Any], keep_last: int = 5) -> None:
    """Append a history entry to the session, capped at 5 most recent turns."""
    sess = get_session(session_id)
    hist = sess.get("history")
    if not isinstance(hist, list):
        hist = []
    hist.append(entry)
    if len(hist) > keep_last:
        hist = hist[-keep_last:]
    sess["history"] = hist
    sess["updated_at"] = now_iso()


def clear_session(session_id: str) -> bool:
    if session_id in SESSIONS:
        del SESSIONS[session_id]
        return True
    return False

def clear_all_sessions() -> None:
    SESSIONS.clear()

def get_all_sessions() -> Dict[str, Dict[str, Any]]:
    return SESSIONS