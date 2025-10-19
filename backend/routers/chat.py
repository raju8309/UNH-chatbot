import re
from typing import Optional
from fastapi import APIRouter, Header, HTTPException
from general_responses import get_generic_response
from models.api_models import ChatRequest, ChatResponse
from services.intent_service import (
    detect_intent,
    detect_program_level,
    get_intent_template
)
from services.qa_service import cached_answer_with_path
from services.session_service import get_session, update_session, clear_session, clear_all_sessions
from utils.course_utils import detect_course_code, COURSE_CODE_RX
from utils.logging_utils import log_chat_interaction
from utils.program_utils import match_program_alias

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def answer_question(
    request: ChatRequest,
    x_session_id: Optional[str] = Header(default=None)
):
    if not x_session_id:
        raise HTTPException(
            status_code=400,
            detail="Please include X-Session-Id header"
        )
    
    sess = get_session(x_session_id)
    
    # handle list messages
    incoming_message = (
        request.message if not isinstance(request.message, list)
        else " ".join(request.message)
    )
    
    # check for generic responses
    resp = get_generic_response(incoming_message)
    if resp:
        return ChatResponse(answer=resp, sources=[], retrieval_path=[])
    
    # update session context
    new_intent = detect_intent(incoming_message, prev_intent=sess.get("intent"))
    new_level = detect_program_level(
        incoming_message,
        fallback=sess.get("program_level") or "unknown"
    )
    
    match = match_program_alias(incoming_message)
    new_alias = match or sess.get("program_alias")
    
    # detect course code
    detected_course = detect_course_code(incoming_message)
    
    # follow-up pattern for courses
    if not detected_course and sess.get("intent") == "course_info":
        if re.search(r"\bwhat about\b", incoming_message, re.I) or \
           COURSE_CODE_RX.search(incoming_message.upper()):
            detected_course = detect_course_code(incoming_message)
    
    # save session updates
    update_session(
        x_session_id,
        intent=new_intent,
        program_level=new_level,
        program_alias=new_alias,
        course_code=detected_course,
        last_question=incoming_message,
    )
    
    # compose scoped question
    scoped_message = incoming_message
    alias_url = None
    
    if new_alias and isinstance(new_alias, dict):
        alias_url = new_alias.get("url")
    
    intent_key = new_intent or sess.get("intent")
    
    # course-specific scoping
    if detected_course and intent_key == "course_info":
        scoped_message = f"course details and prerequisites for {detected_course['norm']}"
    
    # program-specific scoping
    elif new_alias and intent_key:
        template = get_intent_template(intent_key)
        if template:
            prog_title = new_alias["title"]
            scoped_message = f"{template} for {prog_title}"
            if intent_key != "degree_credits" and new_level and new_level != "unknown":
                scoped_message += f" ({new_level})"
    
    # retrieve and answer
    course_norm = detected_course["norm"] if detected_course else None
    answer, sources, retrieval_path = cached_answer_with_path(
        scoped_message,
        alias_url=alias_url,
        intent_key=intent_key,
        course_norm=course_norm
    )
    
    # update session with results
    update_session(
        x_session_id,
        last_answer=answer,
        last_retrieval_path=retrieval_path,
    )
    
    # log interaction
    log_chat_interaction(incoming_message, answer, sources)
    
    return ChatResponse(answer=answer, sources=sources, retrieval_path=retrieval_path)

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