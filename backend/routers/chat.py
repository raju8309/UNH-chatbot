from typing import Optional
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
from general_responses import get_generic_response
from models.api_models import ChatRequest, ChatResponse
from services.session_service import (
    update_session, clear_session, clear_all_sessions,
    push_history, now_iso
)
from utils.logging_utils import log_chat_interaction, log_answer_selection
from services.query_pipeline import process_question_for_retrieval

router = APIRouter()

class AnswerSelectionLog(BaseModel):
    selected_version: str
    answer_mode: str
    timestamp: str
    question: Optional[str] = None
    primary_answer: Optional[str] = None
    alternative_answer: Optional[str] = None
    gold_similarity: Optional[float] = None

@router.post("/chat", response_model=ChatResponse)
async def answer_question(request: ChatRequest, x_session_id: Optional[str] = Header(default=None)):
    if not x_session_id:
        raise HTTPException(
            status_code=400,
            detail="Please include X-Session-Id header"
        )

    incoming_message = request.message if not isinstance(request.message, list) else " ".join(request.message)

    # check for generic responses first
    resp = get_generic_response(incoming_message)
    if resp:
        return ChatResponse(answer=resp, sources=[], retrieval_path=[])

    # process question - now includes dual answer support
    result = process_question_for_retrieval(incoming_message)
    answer = result["answer"]

    # update session with last question
    update_session(
        x_session_id,
        last_question=incoming_message,
        last_answer=answer
    )

    # store simplified history
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

    # build response with all dual answer fields
    response_data = {
        "answer": answer,
        "sources": result["sources"],
        "retrieval_path": result["retrieval_path"],
        "transformed_query": result.get("transformed_query"),
        "has_alternative": result.get("has_alternative", False),
        "alternative_answer": result.get("alternative_answer"),
        "alternative_sources": result.get("alternative_sources"),
        "alternative_retrieval_path": result.get("alternative_retrieval_path"),
        "alternative_type": result.get("alternative_type"),
        "answer_mode": result.get("answer_mode", "retrieval_only"),
        "direct_gold_match": result.get("direct_gold_match", False),
        "gold_similarity": result.get("gold_similarity")
    }

    return ChatResponse(**response_data)

@router.post("/log-answer-selection")
async def log_selection(
    selection: AnswerSelectionLog,
    x_session_id: Optional[str] = Header(default=None)
):
    if not x_session_id:
        raise HTTPException(
            status_code=400,
            detail="Please include X-Session-Id header"
        )
    
    try:
        # log the selection to CSV with all fields
        log_answer_selection(
            session_id=x_session_id,
            selected_version=selection.selected_version,
            answer_mode=selection.answer_mode,
            timestamp=selection.timestamp,
            question=selection.question,
            primary_answer=selection.primary_answer,
            alternative_answer=selection.alternative_answer,
            gold_similarity=selection.gold_similarity
        )
        
        return {
            "status": "logged",
            "session_id": x_session_id,
            "selected_version": selection.selected_version,
            "answer_mode": selection.answer_mode
        }
    except Exception as e:
        print(f"Error logging answer selection: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

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