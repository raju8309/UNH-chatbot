import re
from typing import Optional
from utils.course_utils import detect_course_code, COURSE_CODE_RX

# intent keyword mappings
INTENT_KEYWORDS = {
    "gpa_minimum": [
        "good standing", "minimum gpa", "stay in good standing",
        "probation", "dismissal"
    ],
    "admissions": [
        "admission requirement", "admissions", "apply",
        "application deadline", "recommendation letters", "letter of recommendation",
        "toefl", "ielts", "english proficiency", "gre", "gmat"
    ],
    "credit_transfer": [
        "transfer credit", "transfer credits", "external to unh", "internal to unh"
    ],
    "registration": [
        "add drop", "withdraw", "last day to drop", "registration", "withdrawal"
    ],
    "course_info": [
        "prerequisite", "syllabus", "course description"
    ],
    "degree_credits": [
        "how many credits", "total credits", "credit requirement",
        "number of credits", "credit hours", "credits required"
    ],
}

# intent templates for scoped queries
INTENT_TEMPLATES = {
    "gpa_minimum": "minimum GPA to stay in good standing",
    "admissions": "admission requirements",
    "credit_transfer": "transfer credit policy",
    "registration": "add/drop and withdrawal deadlines",
    "course_info": "course details and prerequisites",
    "degree_credits": "total credits required",
}

# level detection hints
LEVEL_HINTS = {
    "undergrad": ["undergrad", "bachelor", "bs", "ba"],
    "grad": ["graduate", "grad", "master", "ms", "m.s.", "ma", "m.a."],
    "phd": ["phd", "ph.d.", "doctoral", "doctorate"],
    "certificate": ["certificate", "grad certificate", "graduate certificate"],
}

# follow-up detection hints
FOLLOWUP_HINTS = (
    "for ", "now ", "do it for", "for the", "make that for",
    "do that for", "do it", "that", "this", "same"
)

def normalize_query(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def looks_like_followup(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    
    # very short or starts with common follow-up phrases
    return (
        (len(t) <= 60 and any(t.startswith(h) for h in FOLLOWUP_HINTS))
        or t in ("same", "that", "this")
    )

def detect_intent(message: str, prev_intent: Optional[str] = None) -> Optional[str]:
    q = normalize_query(message)
    
    # sticky intent on likely follow-ups
    if looks_like_followup(message) and prev_intent:
        return prev_intent
    
    # find best matching intent
    best = prev_intent
    best_hits = 0
    
    for intent, keywords in INTENT_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in q)
        if hits > best_hits:
            best_hits = hits
            best = intent
    
    # course code detection overrides to course_info
    if detect_course_code(message):
        best = "course_info"
    
    return best

def detect_program_level(message: str, fallback: str = "unknown") -> str:
    q = normalize_query(message)
    
    # check for course code level
    m = COURSE_CODE_RX.search(message or "")
    if m:
        try:
            num = int(re.findall(r"\d+", m.group(2))[0])
            if num >= 900:
                return "phd"
            if num >= 800:
                return "grad"
        except Exception:
            pass
    
    # check for level keywords
    for level, hints in LEVEL_HINTS.items():
        if any(h in q for h in hints):
            return level
    
    return fallback

def get_intent_template(intent_key: Optional[str]) -> Optional[str]:
    return INTENT_TEMPLATES.get(intent_key) if intent_key else None