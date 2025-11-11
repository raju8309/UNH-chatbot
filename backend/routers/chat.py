from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from general_responses import get_generic_response
from models.api_models import ChatRequest, ChatResponse
from services.session_service import update_session, clear_session, clear_all_sessions, push_history, now_iso
from utils.logging_utils import log_chat_interaction
from services.query_pipeline import process_question_for_retrieval
router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def answer_question(request: ChatRequest, x_session_id: Optional[str] = Header(default=None)):
    if not x_session_id:
        raise HTTPException(
            status_code=400,
            detail="Please include X-Session-Id header"
        )
    incoming_message = request.message if not isinstance(request.message, list) else " ".join(request.message)
    
    # Check for generic responses first
    resp = get_generic_response(incoming_message)
    if resp:
        return ChatResponse(answer=resp, sources=[], retrieval_path=[])
    
    # Process question
    result = process_question_for_retrieval(incoming_message)
    answer = result["answer"]
    
    # Update session with last question
    update_session(
        x_session_id,
        last_question=incoming_message,
        last_answer=answer
    )
    
    # Store simplified history
    try:
        push_history(
            x_session_id,
            {
                "timestamp": now_iso(),
                "question": incoming_message,
                "answer": answer,
                "retrieval_path": result["retrieval_path"],
            },
        )
    except Exception:
        pass
    
    log_chat_interaction(incoming_message, answer, result["sources"])
    return ChatResponse(
        answer=answer, 
        sources=result["sources"], 
        retrieval_path=result["retrieval_path"],
        transformed_query=result["transformed_query"]
    )

@router.post("/reset-session")
def reset_one_session(x_session_id: Optional[str] = Header(default=None)):
    if not x_session_id:
        raise HTTPException(
            status_code=400,
            detail="Please include X-Session-Id header"
        )
    if clear_session(x_session_id):
        return {"status": "session_cleared", "session_id": x_session_id}
    return {"status": "no_session_to_clear", "session_id": x_session_id}

@router.post("/reset")
async def reset_all_sessions():
    clear_all_sessions()
    return {"status": "cleared"}