import re
from typing import Optional, Dict, List, Any, Tuple
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
        "number of credits", "credit hours", "credits required",
        "total number of credits", "credits in total", "credit count"
    ],
    "degree_requirements": [
        "degree requirements", "program requirements", "requirements for the degree"
    ],
    "program_options": [
        "thesis", "non-thesis", "project option", "project", "exam option", "comprehensive exam"
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
    "program_options": "thesis vs project/exam options",
    "degree_requirements": "degree requirements",
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

LEVEL_HINT_TOKEN = {"phd": "ph.d.", "grad": "master's", "certificate": "certificate", "undergrad": "bachelor"}

PROGRAM_PAGES: List[Dict[str, str]] = []

SECTION_STOPWORDS = tuple([
    "upon completion", "program learning outcomes", "admission requirements",
    "requirements", "application requirements", "core courses", "electives",
    "overview", "policies", "sample", "plan of study"
])

def normalize_query(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def looks_like_followup(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    if (len(t) <= 80 and any(t.startswith(h) for h in FOLLOWUP_HINTS)) or t in ("same", "that", "this"):
        return True
    if re.search(r"\b(this|that)\s+(program|course)\b", t):
        return True
    return False

def detect_intent(message: str, prev_intent: Optional[str]) -> Optional[str]:
    q = normalize_query(message)
    if looks_like_followup(message):
        if any(tok in q for tok in ["credit", "credits", "credit hour", "credit hours", "how many", "total number"]):
            return "degree_credits"
        if any(tok in q for tok in ["degree requirement", "degree requirements", "program requirements", "requirements for the degree", "requirements"]):
            return "degree_requirements"
        if detect_course_code(message):
            return "course_info"
        if prev_intent:
            return prev_intent

    best = prev_intent
    best_hits = 0
    for intent, kws in INTENT_KEYWORDS.items():
        hits = sum(1 for kw in kws if kw in q)
        if hits > best_hits:
            best_hits = hits
            best = intent
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
    
    return fallback or "unknown"

def get_intent_template(intent_key: Optional[str]) -> Optional[str]:
    return INTENT_TEMPLATES.get(intent_key) if intent_key else None

def format_followup_answer(answer: str, sess: Dict[str, Any], intent_key: Optional[str], is_followup: bool) -> str:
    text = (answer or "").strip()
    from services.retrieval_service import UNKNOWN
    if not text or text.lower() == UNKNOWN.lower():
        return answer or ""

    try:
        hist = sess.get("history") if isinstance(sess, dict) else None
        if not is_followup or not (isinstance(hist, list) and len(hist) > 0):
            return answer or ""
    except Exception:
        return answer or ""

    label = None
    course = sess.get("course_code") if isinstance(sess, dict) else None
    if isinstance(course, dict) and course.get("norm"):
        label = course["norm"]
    alias = sess.get("program_alias") if isinstance(sess, dict) else None
    if label is None and isinstance(alias, dict) and alias.get("title"):
        label = (alias["title"].split(" - ")[0] or alias["title"]).strip()
    if not label:
        return answer

    try:
        parts = [p.strip() for p in re.split(r"\s*-\s*", label) if p.strip()]
        if len(parts) == 2 and parts[0].lower() == parts[1].lower():
            label = parts[0]
    except Exception:
        pass

    body = text.lstrip()
    if body.lower().startswith(f"for {label.lower()}"):
        return answer
    if body.lower().startswith("for "):
        try:
            head = body[4:].split(",", 1)[0].strip().lower()
            if head == label.lower():
                return answer
        except Exception:
            pass

    first = body[:1]
    rest = body[1:]
    if first.isupper() and (rest[:1].islower() if rest else False) and not body.startswith("I "):
        body = first.lower() + rest

    return f"For {label}, {body}"


def explicit_program_mention(text: str) -> bool:
    t = (text or "").lower()
    has_degree_word = bool(re.search(r"\b(ms|m\.s\.|master'?s|phd|ph\.d\.|doctoral|doctorate|certificate)\b", t))
    has_scope_word = (" program " in f" {t} ") or bool(re.search(r"\b(in|for)\b", t))
    return has_degree_word and has_scope_word


def auto_intent_from_topic(message: str) -> Optional[str]:
    q = normalize_query(message)
    if not q:
        return None
    if any(tok in q for tok in ["good standing", "probation", "dismissal", "minimum gpa", "gpa", "grade", "c grade", "b-", "b minus", "c-", "c minus"]):
        return "gpa_minimum"
    if any(tok in q for tok in ["withdraw", "withdrawal", "drop", "add drop", "deadline", "last day to"]):
        return "registration"
    if any(tok in q for tok in ["credits required", "how many credits", "total credits", "credit requirement", "credit hours"]):
        return "degree_credits"
    if any(tok in q for tok in ["thesis", "non thesis", "non-thesis", "project", "exam option", "comprehensive exam"]):
        return "program_options"
    if detect_course_code(message):
        return "course_info"
    if any(tok in q for tok in ["gre", "gmat", "toefl", "ielts", "admission", "admissions", "apply", "recommendation"]):
        return "admissions"
    return None

def detect_correction_or_negation(text: str) -> Dict[str, Optional[str]]:
    t = (text or "").lower()
    result = {"negated_level": None, "new_level": None}
    for lvl, hints in LEVEL_HINTS.items():
        for h in hints:
            if f"not {h}" in t or re.search(rf"not in (a|the)?\s*{h}", t):
                result["negated_level"] = lvl
    for lvl, hints in LEVEL_HINTS.items():
        for h in hints:
            if re.search(rf"\bfor (a|the)?\s*{h}\b", t) or re.search(rf"i'?m in (a|the)?\s*{h}\b", t):
                result["new_level"] = lvl
    if "actually" in t or "instead" in t:
        for lvl, hints in LEVEL_HINTS.items():
            if any(h in t for h in hints):
                result["new_level"] = lvl
    return result

# --- Policy term detector
def has_policy_terms(text: str) -> bool:
    t = (text or "").lower()
    return any(kw in t for kw in [
        "gpa", "good standing", "probation", "dismissal", "grade", "b-", "b minus", "c grade", "c-", "c minus", "minimum gpa",
        "committee", "guidance committee", "supervisory committee", "qualifying exam", "qualifying examination",
        "final exam", "exam attempt", "examination attempt"
    ])

# --- Admissions term detector and URL helpers ---
def has_admissions_terms(text: str) -> bool:
    t = (text or "").lower()
    return any(kw in t for kw in [
        "admission", "admissions", "apply", "application", "requirements", "gre", "gmat",
        "test score", "test scores", "english proficiency", "toefl", "ielts",
        "letters of recommendation", "recommendation letter"
    ])

def is_admissions_url(url: str) -> bool:
    return isinstance(url, str) and "/graduate/general-information/admissions/" in url

def is_degree_requirements_url(url: str) -> bool:
    return isinstance(url, str) and "/graduate/academic-regulations-degree-requirements/degree-requirements/" in url

def _degree_flags_from_title_url(title: str, url: str) -> Tuple[bool, bool, bool, bool]:
    title_l = (title or "").lower()
    url_l = (url or "").lower()
    is_cert = ("certificate" in title_l) or ("certificate" in url_l)
    is_phd  = ("phd" in title_l) or ("ph.d" in title_l) or ("/phd" in url_l)
    is_ms   = (
        "(m.s" in title_l or " m.s" in title_l or " m.s." in title_l or " ms" in title_l or
        "/ms" in url_l or "-ms/" in url_l or url_l.endswith("-ms/") or "-ms#" in url_l
    )
    is_ug   = ("b.s" in title_l) or ("b.a" in title_l) or ("/bs" in url_l) or ("/ba" in url_l)
    return is_ms, is_phd, is_cert, is_ug

def alias_conflicts_with_level(alias: Optional[Dict[str, str]], level: str) -> bool:
    if not alias or not level or level == "unknown":
        return False
    is_ms, is_phd, is_cert, is_ug = _degree_flags_from_title_url(alias.get("title", ""), alias.get("url", ""))
    if level == "certificate":
        return not is_cert
    if level == "phd":
        return not is_phd
    if level == "grad":
        return not (is_ms or is_phd)
    if level == "undergrad":
        return not is_ug
    return False
