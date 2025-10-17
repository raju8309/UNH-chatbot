from fastapi import APIRouter, HTTPException
from services.chunk_service import get_tier_counts
from services.session_service import get_all_sessions

router = APIRouter()
@router.get("/debug/tier-counts")
def tier_counts():
    counts = get_tier_counts()
    
    return {
        "tier1": counts[1],
        "tier2": counts[2],
        "tier3": counts[3],
        "tier4": counts[4],
        "total": sum(counts.values()),
    }

@router.get("/session/{session_id}")
def get_session_debug(session_id: str):
    sessions = get_all_sessions()
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return sessions[session_id]

@router.get("/sessions")
def get_all_sessions_debug():
    return get_all_sessions()