from typing import Optional
from fastapi import APIRouter, Header, HTTPException
from general_responses import get_generic_response
from models.api_models import ChatRequest, ChatResponse
from services.session_service import get_session, update_session, clear_session, clear_all_sessions, push_history, now_iso
from utils.logging_utils import log_chat_interaction
from services.query_pipeline import process_question_for_retrieval
from services.intent_service import format_followup_answer, looks_like_followup

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def answer_question(request: ChatRequest, x_session_id: Optional[str] = Header(default=None)):
    if not x_session_id:
        raise HTTPException(
            status_code=400,
            detail="Please include X-Session-Id header"
        )
    sess = get_session(x_session_id)
    incoming_message = request.message if not isinstance(request.message, list) else " ".join(request.message)
    resp = get_generic_response(incoming_message)
    if resp:
        return ChatResponse(answer=resp, sources=[], retrieval_path=[])
    result = process_question_for_retrieval(
        incoming_message,
        session=sess,
        prev_intent=sess.get("intent"),
        prev_program_level=sess.get("program_level"),
        prev_program_alias=sess.get("program_alias"),
        prev_course_code=sess.get("course_code"),
        prev_last_question=sess.get("last_question"),
        prev_last_answer=sess.get("last_answer"),
        prev_last_retrieval_path=sess.get("last_retrieval_path")
    )
    # Format follow-up answers for chat only
    is_followup = looks_like_followup(request.message)
    answer = result["answer"]
    try:
        answer = format_followup_answer(answer, sess, result["intent"], is_followup)
    except Exception:
        pass
    update_session(
        x_session_id,
        **result["session_updates"]
    )
    try:
        push_history(
            x_session_id,
            {
                "timestamp": now_iso(),
                "question": result["session_updates"].get("last_question"),
                "scoped_message": result["scoped_message"],
                "answer": answer,
                "intent": result["intent"],
                "program_level": result["program_level"],
                "program_alias": (result["program_alias"] or {}).get("title") if isinstance(result["program_alias"], dict) else None,
                "course_code": (result["course_code"] or {}).get("norm") if isinstance(result["course_code"], dict) else None,
                "retrieval_path": result["retrieval_path"],
            },
        )
    except Exception:
        pass
    log_chat_interaction(result["session_updates"].get("last_question"), answer, result["sources"])
    return ChatResponse(answer=answer, sources=result["sources"], retrieval_path=result["retrieval_path"])

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