import re
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote_plus

# regex patterns
COURSE_CODE_RX = re.compile(r"\b([A-Z]{2,5})\s*-?\s*(\d{3}[A-Z]?)\b")
COURSE_LINE_CREDITS_RX = re.compile(r"\bCredits:\s*([0-9]+)\b", re.I)
COURSE_LINE_PREREQ_RX = re.compile(r"\bPrerequisite\(s\):\s*(.+?)(?:\.\s*$|$)", re.I)
COURSE_LINE_GRADEMODE_RX = re.compile(r"\bGrade\s*Mode:\s*([A-Za-z/ \-]+)$", re.I)

def detect_course_code(message: str) -> Optional[Dict[str, str]]:
    if not message:
        return None
    
    m = COURSE_CODE_RX.search(message.upper())
    if not m:
        return None
    
    subj = re.sub(r"[^A-Z]", "", m.group(1).upper())
    num = m.group(2).upper().replace("-", "").replace(" ", "")
    
    return {
        "subj": subj,
        "num": num,
        "norm": f"{subj} {num}"
    }

def get_course_search_url(norm: str) -> str:
    return f"https://catalog.unh.edu/search/?P={quote_plus(norm)}"

def url_contains_course(url: str, norm: str) -> bool:
    if not url:
        return False
    
    u = url.lower()
    encoded_space = norm.lower().replace(" ", "%20")
    plus_space = norm.lower().replace(" ", "+")
    raw = norm.lower()
    
    return (
        f"/search/?p={encoded_space}" in u or
        f"/search/?p={plus_space}" in u or
        f"/search/?p={raw}" in u
    )

def title_starts_with_course(title: str, norm: str) -> bool:
    t = (title or "").upper().strip()
    return t.startswith(norm.upper())

def extract_course_fallbacks(chunks: List[Tuple[str, Dict]]) -> Dict[str, Optional[str]]:
    result = {
        "credits": None,
        "prereqs": None,
        "grademode": None
    }
    
    for text, _ in chunks:
        if not text:
            continue
        
        # extract credits
        if result["credits"] is None:
            m = COURSE_LINE_CREDITS_RX.search(text)
            if m:
                result["credits"] = m.group(1)
        
        # extract prerequisites
        if result["prereqs"] is None:
            m = COURSE_LINE_PREREQ_RX.search(text)
            if m:
                result["prereqs"] = m.group(1).strip()
        
        # extract grade mode
        if result["grademode"] is None:
            m = COURSE_LINE_GRADEMODE_RX.search(text)
            if m:
                result["grademode"] = m.group(1).strip()
    
    return result