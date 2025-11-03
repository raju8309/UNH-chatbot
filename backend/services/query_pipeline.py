import re

# simple external-link detector for calendar/deadline queries ===
_CALENDAR_LINK = "https://www.unh.edu/registrar/registration-resources/calendars-important-deadlines"
_CALENDAR_KEYWORDS = {
    "academic calendar", "calendar", "important dates", "important deadlines",
    "deadlines", "deadline", "add/drop", "add drop", "drop deadline", "withdraw deadline",
    "registration deadline", "semester start", "semester end", "term start", "term end",
    "holiday", "break", "vacation", "last day to add", "last day to drop"
}
def _maybe_calendar_link(message: str):
    q = (message or "").lower()
    if any(k in q for k in _CALENDAR_KEYWORDS):
        return (
            f"For up-to-date academic dates and deadlines, see the Registrar’s "
            f"calendars and important deadlines: {_CALENDAR_LINK}"
        )
    return None

from services.intent_service import (
    LEVEL_HINT_TOKEN,
    INTENT_TEMPLATES,
    detect_intent,
    detect_program_level,
    detect_correction_or_negation,
    alias_conflicts_with_level,
    looks_like_followup,
    explicit_program_mention,
    auto_intent_from_topic
)
from services.qa_service import cached_answer_with_path
from utils.course_utils import detect_course_code, COURSE_CODE_RX
from utils.program_utils import match_program_alias
from services.query_transform_service import transform_query

def process_question_for_retrieval(
    incoming_message,
    session=None,
    prev_intent=None,
    prev_program_level=None,
    prev_program_alias=None,
    prev_course_code=None,
    prev_last_question=None,
    prev_last_answer=None,
    prev_last_retrieval_path=None
):
    # session or a fake session
    sess = session or {}
    sess.setdefault("intent", prev_intent)
    sess.setdefault("program_level", prev_program_level)
    sess.setdefault("program_alias", prev_program_alias)
    sess.setdefault("course_code", prev_course_code)
    sess.setdefault("last_question", prev_last_question)
    sess.setdefault("last_answer", prev_last_answer)
    sess.setdefault("last_retrieval_path", prev_last_retrieval_path)

    # handle list messages
    if isinstance(incoming_message, list):
        incoming_message = " ".join(incoming_message)

    # Apply query transformation before intent detection and retrieval
    user_query = incoming_message
    _transformed = transform_query(user_query)
    if _transformed != user_query:
        print(f"[QueryTransform] Original: {user_query} -> Transformed: {_transformed}")
    user_query = _transformed

    # update session context
    new_intent = detect_intent(user_query, prev_intent=sess.get("intent"))
    new_level = detect_program_level(
        user_query,
        fallback=sess.get("program_level") or "unknown"
    )
    match = match_program_alias(user_query)
    new_alias = match or sess.get("program_alias")

    corr = detect_correction_or_negation(user_query)
    if corr.get("negated_level"):
        neg = corr["negated_level"]
        if sess.get("program_level") == neg:
            new_level = "unknown"
            new_alias = None
    if corr.get("new_level"):
        new_level = corr["new_level"]
    if any(corr.values()):
        if sess.get("last_question"):
            user_query = sess.get("last_question")

    try:
        if new_alias and isinstance(new_alias, dict) and new_level and new_level != "unknown":
            if alias_conflicts_with_level(new_alias, new_level):
                level_hint = LEVEL_HINT_TOKEN.get(new_level, "")
                hinted_message = user_query + (f" {level_hint}" if level_hint else "")
                rematch = match_program_alias(hinted_message)
                new_alias = rematch if rematch else None
    except Exception:
        pass

    if isinstance(new_alias, dict) and new_alias.get("title"):
        new_alias = {
            "title": (new_alias["title"].split(" - ")[0] or new_alias["title"]).strip(),
            "url": new_alias.get("url", ""),
        }

    is_followup = looks_like_followup(user_query) or any(corr.values())
    base_topic = user_query
    if is_followup and sess.get("last_question"):
        base_topic = sess.get("last_question")

    try:
        explicit_prog = explicit_program_mention(user_query)

        if is_followup and sess.get("program_alias") and not explicit_prog:
            new_alias = sess.get("program_alias")
        elif match:
            new_alias = match
        else:
            new_alias = sess.get("program_alias") if explicit_prog else None
    except Exception:
        pass

    if not new_intent:
        inferred = auto_intent_from_topic(base_topic)
        if inferred:
            new_intent = inferred
        elif sess.get("intent"):
            new_intent = sess.get("intent")

    detected_course = detect_course_code(base_topic)
    if not detected_course and (sess.get("intent") == "course_info"):
        if re.search(r"\bwhat about\b", user_query, re.I) or COURSE_CODE_RX.search(user_query.upper()):
            detected_course = detect_course_code(user_query)

    # session updates
    session_updates = dict(
        intent=new_intent,
        program_level=new_level,
        program_alias=new_alias,
        course_code=detected_course,
        last_question=base_topic,
    )

    scoped_message = base_topic
    alias_url = None
    if new_alias and isinstance(new_alias, dict):
        alias_url = new_alias.get("url")

    # Not using scoped_message, intent_key, or course_norm as they all tank the test answers/scores
    answer, sources, retrieval_path, context = cached_answer_with_path(
        user_query, alias_url=alias_url, intent_key=None, course_norm=None
    )

    return dict(
        answer=answer,
        sources=sources,
        retrieval_path=retrieval_path,
        session_updates=session_updates,
        context=context,
        intent=None,
        program_level=new_level,
        program_alias=new_alias,
        course_code=None,
        scoped_message=scoped_message,
    )