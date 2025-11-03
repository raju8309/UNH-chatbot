# backend/services/query_transform_service.py
import re
import json
import os
from pathlib import Path

import importlib

# Resolve a callable text-generation function from models/ml_models.py with several fallbacks.
_CALL_FN = None
try:
    ml_module = importlib.import_module("models.ml_models")
    for _name in ("call_model", "generate_text", "generate", "complete", "run", "infer", "invoke", "chat", "predict"):
        _candidate = getattr(ml_module, _name, None)
        if callable(_candidate):
            _CALL_FN = _candidate
            break
    if _CALL_FN is None:
        raise ImportError("No callable text-generation function found in models/ml_models.py (tried: call_model, generate_text, generate, complete, run, infer, invoke, chat, predict)")
except Exception as _e:
    # Defer raising until use so the module can still be imported for rule-only mode
    _IMPORT_ERROR = _e
else:
    _IMPORT_ERROR = None

from config import settings

# Resolve repo root and default rules path (../config/query_rewrite.json)
ROOT_DIR = Path(__file__).resolve().parents[1]
RULES_PATH = ROOT_DIR / "config" / "query_rewrite.json"

def _load_domain_rules():
    if RULES_PATH.exists():
        try:
            return json.loads(RULES_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    # Safe defaults if file missing
    return {
        "failing": "failing grade",
        "transfer": "transfer credit",
        "capstone": "capstone requirement",
        "time": "time limit for degree completion",
        "masters": "master’s program",
        "phd": "Ph.D. program",
        "gpa": "minimum GPA requirement"
    }

DOMAIN_RULES = _load_domain_rules()

COURSE_CODE_RE = re.compile(r"\b[A-Z]{2,4}\s?\d{3}[A-Z]?\b")
NUMBER_RE      = re.compile(r"\d+")

def normalize_text(text: str) -> str:
    # keep case for entities later; do light cleanup only
    t = text.strip()
    t = re.sub(r"\s+", " ", t)
    return t

def apply_domain_rules(text: str) -> str:
    out = text
    for key, val in DOMAIN_RULES.items():
        pattern = re.compile(rf"\b{re.escape(key)}\b", flags=re.IGNORECASE)
        # Skip replacement if:
        # 1. The expanded form is already present (e.g., "academic probation" already there)
        # 2. The key word appears adjacent to words from the expansion (e.g., "GPA requirement" + rule "gpa"->"minimum GPA requirement")
        val_lower = val.lower()
        out_lower = out.lower()
        
        # Check if expansion already exists
        if val_lower in out_lower:
            continue
            
        # Check if key appears in a context that overlaps with the expansion
        # e.g., "GPA requirement" shouldn't become "minimum GPA requirement requirement"
        val_words = set(val_lower.split())
        matches = pattern.finditer(out_lower)
        skip = False
        for match in matches:
            start, end = match.span()
            # Check words immediately before and after the match
            context_start = max(0, start - 30)
            context_end = min(len(out_lower), end + 30)
            context = out_lower[context_start:context_end]
            context_words = set(context.split())
            # If other words from the expansion are already nearby, skip
            if len(val_words.intersection(context_words) - {key.lower()}) > 0:
                skip = True
                break
        
        if not skip:
            out = pattern.sub(val, out)
    return out

def _extract_entities(text: str):
    courses = set(COURSE_CODE_RE.findall(text))
    numbers = set(NUMBER_RE.findall(text))
    return courses, numbers

def validate_rewrite(original: str, rewritten: str) -> bool:
    # 1) must not drop course codes
    orig_courses, orig_nums = _extract_entities(original)
    new_courses,  new_nums  = _extract_entities(rewritten)

    if not orig_courses.issubset(new_courses):
        return False

    # 2) must not introduce new numeric facts (allow subset or equal)
    if not new_nums.issuperset(orig_nums):
        return False

    # 3) keep it short and useful
    if len(rewritten.split()) < 3:
        return False

    return True

# --- Semantic sanity helpers ---
_KEY_TOKEN_HINTS = tuple({
    # program/level
    "phd", "ph.d", "doctoral", "doctorate", "master", "masters", "m.s.", "ms", "m.a.", "ma", "certificate",
    # caps/thesis/project/exams
    "thesis", "non thesis", "non-thesis", "project", "capstone", "comprehensive exam", "comps", "exam option",
    # grading/policies
    "gpa", "grade", "b-", "pass/fail", "pass fail", "ia", "audit",
    # logistics/rules
    "time limit", "deadline", "withdraw", "withdrawal", "repeat", "prereq", "prereqs", "prerequisite", "transfer",
    # credits
    "credit", "credits",
})

def _extract_key_tokens(text: str) -> set:
    t = (text or "").lower()
    hits = set()
    for tok in _KEY_TOKEN_HINTS:
        if tok in t:
            hits.add(tok)
    return hits

def _semantic_sanity(original: str, rewritten: str) -> bool:
    """
    Reject rewrites that are repetitive/irrelevant.
    - Repeats of certain phrases more than once (e.g., 'transfer credit', 'UNH Graduate Catalog').
    - No overlap with key tokens present in the original.
    """
    o = (original or "")
    r = (rewritten or "")
    rl = r.lower()

    # repetition guard
    for phrase in ("transfer credit", "unh graduate catalog"):
        if len(re.findall(re.escape(phrase), rl, flags=re.IGNORECASE)) > 1:
            return False

    # key token overlap guard
    orig_keys = _extract_key_tokens(o)
    if orig_keys:
        if not any(k in rl for k in orig_keys):
            return False

    return True

def llm_rewrite(query: str) -> str:
    """
    Use the existing model to produce a precise, catalog-specific rewrite.
    Assumes call_model(prompt: str) -> str.
    """
    prompt = (
        "You rewrite user questions to be precise and UNH Graduate Catalog-specific.\n"
        "Rules:\n"
        "- Clarify vague nouns (e.g., 'failing' -> 'failing grade').\n"
        "- Do NOT invent facts, numbers, or policies.\n"
        "- Preserve all course codes and entities.\n"
        "- Keep it to a single concise question.\n\n"
        "Examples:\n"
        "User: what is failing\n"
        "Rewritten: What is a failing grade in the UNH Graduate Catalog?\n\n"
        "User: thesis vs non thesis\n"
        "Rewritten: What is the difference between thesis and non-thesis options in UNH graduate programs?\n\n"
        "User: capstone?\n"
        "Rewritten: What is the capstone requirement for UNH master’s programs?\n\n"
        "User: audit option?\n"
        "Rewritten: Are UNH graduate students allowed to audit courses?\n\n"
        "User: comps required?\n"
        "Rewritten: Are comprehensive exams required for UNH graduate degrees?\n\n"
        "User: time limit masters\n"
        "Rewritten: What is the time limit to complete a UNH master’s degree?\n\n"
        f"User question: {query}\n"
        "Rewritten question:"
    )
    try:
        if _CALL_FN is None:
            if getattr(settings, "ENABLE_QUERY_REWRITER", False):
                # Surface a clearer message if rewrite was explicitly enabled but we couldn't import a model callable
                raise ImportError(f"[QueryRewrite] Could not locate a callable in models/ml_models.py. Original import error: {_IMPORT_ERROR}")
            # If rewriter disabled, we won't call a model anyway.
            return query

        resp = _CALL_FN(prompt)
        rewritten = (resp or "").strip()

        # Token-preservation check: ensure core content words survive the rewrite
        KEY_TOKENS = re.findall(r"[a-zA-Z]+", (query or "").lower())
        STOP = {"what","is","are","the","a","an","in","for","to","of","and","or","at","on","vs","do","does","about","with","from"}
        KEY_TOKENS = [t for t in KEY_TOKENS if t not in STOP]
        if KEY_TOKENS and not any(t in (rewritten or "").lower() for t in KEY_TOKENS):
            return query

        # Semantic sanity check: reject noisy/irrelevant rewrites
        if not _semantic_sanity(query, rewritten):
            return query

        return rewritten
    except Exception as e:
        print(f"[QueryRewrite] call_model failed: {e}")
        return query

def transform_query(user_query: str) -> str:
    """
    Main entry: normalize -> domain rules -> optional LLM rewrite -> validate -> pick best.
    """
    original = user_query
    cleaned  = normalize_text(original)
    expanded = apply_domain_rules(cleaned)

    if getattr(settings, "ENABLE_QUERY_REWRITER", False):
        rewritten = llm_rewrite(expanded)
        if validate_rewrite(original, rewritten):
            return rewritten

    return expanded