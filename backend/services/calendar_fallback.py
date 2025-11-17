
import re
from typing import Optional, Sequence
from config.settings import get_config

# put the registrar calendar in one place for easy updates
REGISTRAR_CAL_URL = "https://www.unh.edu/registrar/registration-resources/calendars-important-deadlines"

# calendar-like intents (dynamic dates that change yearly)
_CALENDAR_KEYWORDS = [
    r"\bdeadline\b", r"\bdeadlines\b", r"\bdue date\b", r"\bdue\b",
    r"\bstart date\b", r"\bend date\b", r"\bwhen (does|do)\b",
    r"\bj[-\s]?term\b", r"\bjanuary term\b", r"\bfinal(s)?\b",
    r"\bexam(s)?\b", r"\bad[d/-]drop\b", r"\bregistration\b",
    r"\bbreak\b", r"\bholiday(s)?\b", r"\bspring\b", r"\bfall\b", r"\bsummer\b",
    r"\bsemester\b", r"\bterm\b", r"\bschedule\b", r"\bcalendar\b"
]

# do not route these — static/policy answers exist
_NEVER_ROUTE_PHRASES = [
    # keep your good policy answers intact:
    "appeal", "dismissal", "academic probation", "probation",
    "candidacy", "residency requirement", "grading policy",
    "thesis defense", "non-thesis option"
]

# date-ish patterns that mean we *did* answer concretely
_DATE_PATTERNS = [
    # Jan 12, 2026 / January 12, 2026
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)\s+\d{1,2},\s+\d{4}\b",
    # 1/12/2026 or 2026-01-12
    r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
    r"\b\d{4}-\d{2}-\d{2}\b",
    # “within 10 business days”, “within 20 days”
    r"\bwithin\s+\d+\s+(business\s+)?days\b",
    # “by May 15”, “due May 15”
    r"\b(?:by|due)\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)\s+\d{1,2}\b",
]

_cal_kw = re.compile("|".join(_CALENDAR_KEYWORDS), re.IGNORECASE)
_never = re.compile("|".join(map(re.escape, _NEVER_ROUTE_PHRASES)), re.IGNORECASE)
_date_ok = [re.compile(p, re.IGNORECASE) for p in _DATE_PATTERNS]

def _looks_calendar_like(question: str) -> bool:
    return bool(_cal_kw.search(question)) and not _never.search(question)

def _answer_has_clear_date(answer: str) -> bool:
    if not answer:
        return False
    return any(p.search(answer) for p in _date_ok)

def _retrieval_had_datey_source(source_titles: Sequence[str]) -> bool:
    """Optionally boost confidence if sources already look date-like."""
    joined = " | ".join(source_titles).lower()
    # if we already linked Registrar calendar or a page named 'January Term' etc., treat as confident
    return ("academic calendar" in joined) or ("january term" in joined) or ("registration - january term" in joined)

def maybe_calendar_fallback(
    question: str,
    generated_answer: str,
    source_titles: Sequence[str] = (),
) -> Optional[str]:
    """
    Return a fallback calendar message iff:
      - calendar linking is enabled in config AND
      - question is calendar-like (dynamic dates) AND
      - the model did NOT produce a clear date AND
      - retrieval didn't already include obviously date-specific sources
    Otherwise return None (keep existing answer).
    """
    # Check if calendar linking is enabled
    cfg = get_config()
    calendar_enabled = cfg.get("calendar_linking", {}).get("enabled", True)
    if not calendar_enabled:
        return None
    
    if not _looks_calendar_like(question):
        return None
    if _answer_has_clear_date(generated_answer):
        return None
    if _retrieval_had_datey_source(source_titles):
        return None

    # polite, non-blocking fallback
    return (
        "Dates can vary by year and program. For the most current official dates "
        f"(start/end of term, add/drop, holidays, finals, etc.), please see the "
        f"Registrar's Academic Calendar: {REGISTRAR_CAL_URL}"
    )