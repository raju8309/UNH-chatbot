# backend/services/query_transform_service.py
import re
import json
from pathlib import Path
import importlib

try:
    # Optional: embedding helper for semantic similarity checks
    from models.ml_models import get_text_embedding
except Exception:
    get_text_embedding = None  # type: ignore

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
        "gpa": "minimum GPA requirement",
        # added enriched defaults
        "add/drop": "add/drop deadline",
        "overload": "overload permission",
        "waiver": "course waiver",
        "double counting": "double-counting",
        "residency": "residency requirement",
        "continuous": "continuous enrollment",
        "audit": "audit option",
        "comps": "comprehensive examination",
        "thesis": "thesis or non-thesis option",
    }

DOMAIN_RULES = _load_domain_rules()

COURSE_CODE_RE = re.compile(r"\b[A-Z]{2,4}\s?\d{3}[A-Z]?\b")
NUMBER_RE      = re.compile(r"\d+")

# Stopwords and guards for token preservation and repetition
STOPWORDS = {"what","is","are","the","a","an","in","for","to","of","and","or","at","on","vs","does","do","can","how","many","long"}
# Phrases we never want to see echoed from the prompt itself
INSTRUCTION_PHRASES = {
    "you rewrite user questions to be precise and unh graduate catalog-specific",
    "rewrite the student question as a single clear question",
    "unh graduate school policies or programs",
}
# Transfer-related boilerplate that should never appear unless the user actually asked about transfer
_TRANSFER_FORBIDDEN_BASE = {
    "external graduate transfer credits",
    "what are the rules for transferring graduate credits from another university to unh",
    "rules for transferring graduate credits from another university to unh",
}
TRANSFER_FORBIDDEN_PHRASES = _TRANSFER_FORBIDDEN_BASE.union(
    {p.lower() for p in getattr(settings, "FORBIDDEN_PHRASES", set())}
)

def _dedupe_adjacent_words(text: str) -> str:
    """
    Collapse immediately repeated words (case-insensitive), e.g.
    'audit option option?' -> 'audit option?'.
    """
    if not text:
        return text
    pattern = re.compile(r"\b(\w+)(\s+\1\b)", flags=re.IGNORECASE)
    out = text
    while True:
        new = pattern.sub(r"\1", out)
        if new == out:
            break
        out = new
    return out

def _content_tokens(text: str) -> set:
    return {t for t in re.findall(r"[a-zA-Z]+", (text or "").lower()) if t not in STOPWORDS}

def _has_dup_words(text: str) -> bool:
    words = re.findall(r"[a-zA-Z]+", (text or "").lower())
    for i in range(len(words)-1):
        if words[i] == words[i+1]:
            return True
    return False

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

# Special-case rewrites for a few frequently asked but hard-to-parse questions
SPECIAL_REWRITES = {
    "final exam attempts masters?": "How many attempts do UNH master’s students have to pass the final exam?",
    "accelerated student below b minus?": "What happens if an accelerated UNH graduate student earns below a B-?",
    "thesis vs non thesis?": "What is the difference between thesis and non-thesis options in UNH graduate programs?",
    "leave of absence grad?": "What is the leave-of-absence policy for UNH graduate students?",
    "phd time limit?": "What is the time limit for completing a UNH Ph.D. program?",
    "masters time limit?": "What is the time limit for completing a UNH master’s degree?",
}

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
    # added tokens
    "add/drop", "overload", "waiver", "residency", "enrollment",
    # status / international / funding
    "probation", "dismissal", "visa", "sevis", "assistantship", "stipend", "insurance", "leave",
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

# --- Intent detection and policy noun helpers ---

POLICY_NOUNS = {
    "transfer", "probation", "dismissal", "assistantship", "visa", "sevis",
    "insurance", "leave", "capstone", "thesis", "enrollment", "deadline",
    # program names
    "biotechnology", "biotech", "cybersecurity", "information technology", "it ms",
}

def _policy_terms(text: str) -> set:
    t = (text or "").lower()
    return {term for term in POLICY_NOUNS if term in t}


def infer_intent(query: str) -> str:
    """Very coarse intent classification for building a focused prompt."""
    q = (query or "").lower()

    if any(w in q for w in ("gpa", "grade", "failing", "pass/fail", "pass fail")):
        return "grading"
    if any(w in q for w in ("time limit", "years", "semesters", "duration", "complete degree", "completion")):
        return "time_limit"
    if any(w in q for w in ("transfer", "double counting", "certificate", "taking courses elsewhere", "external coursework")):
        return "transfer"
    if any(w in q for w in ("leave", "withdraw", "withdrawal", "add/drop", "add drop", "continuous enrollment", "registration", "readmission", "medical withdrawal", "hold")):
        return "enrollment"
    if any(w in q for w in ("assistantship", "ra", "ta", "stipend", "tuition waiver", "funding", "fellowship")):
        return "assistantship"
    if any(w in q for w in ("visa", "sevis", "international", "english proficiency", "toefl", "ielts")):
        return "international"
    # program-specific intents
    if any(w in q for w in ("biotechnology", "biotech")):
        return "program_biotech"
    if any(w in q for w in ("cybersecurity", "cyber security")):
        return "program_cyber"
    if any(w in q for w in ("information technology", "it ms", "it m.s")):
        return "program_it"
    if any(w in q for w in ("grad 900", "independent study", "internship", "jterm", "j-term")):
        return "program_info"
    return "general"


def build_prompt(query: str, intent: str) -> str:
    """Build a small, intent-focused prompt for the rewriter."""
    base = (
        "Rewrite the student question as a single clear question about UNH graduate school policies or programs.\n"
        "Rules:\n"
        "- Keep the same meaning and important words (e.g., add/drop, overload, transfer, visa, probation).\n"
        "- Do not invent any new numbers, requirements, or policy details.\n"
        "- Write one concise sentence.\n\n"
    )

    if intent == "grading":
        examples = (
            "Examples:\n"
            "User: grad gpa for graduation?\n"
            "Rewrite: What GPA do UNH graduate students need in order to graduate?\n\n"
            "User: minimum grade for grad credit?\n"
            "Rewrite: What is the minimum grade required to earn graduate credit at UNH?\n\n"
        )
    elif intent == "time_limit":
        examples = (
            "Examples:\n"
            "User: phd time limit?\n"
            "Rewrite: What is the time limit for completing a UNH Ph.D. program?\n\n"
            "User: masters time limit?\n"
            "Rewrite: What is the time limit for completing a UNH master’s degree?\n\n"
        )
    elif intent == "transfer":
        examples = (
            "Examples:\n"
            "User: transfer credit rules?\n"
            "Rewrite: What are the transfer credit rules for UNH graduate programs?\n\n"
            "User: max transfer credits?\n"
            "Rewrite: What is the maximum number of transfer credits allowed in a UNH graduate program?\n\n"
        )
    elif intent == "enrollment":
        examples = (
            "Examples:\n"
            "User: add drop deadline grad?\n"
            "Rewrite: What is the add/drop deadline for UNH graduate courses?\n\n"
            "User: leave of absence grad?\n"
            "Rewrite: What is the leave-of-absence policy for UNH graduate students?\n\n"
        )
    elif intent == "assistantship":
        examples = (
            "Examples:\n"
            "User: ra ta tuition waiver?\n"
            "Rewrite: Do UNH RA and TA positions include a tuition waiver for graduate students?\n\n"
            "User: assistantship eligibility grad?\n"
            "Rewrite: What are the eligibility requirements for graduate assistantships at UNH?\n\n"
        )
    elif intent == "international":
        examples = (
            "Examples:\n"
            "User: visa full time credits?\n"
            "Rewrite: How many credits are required for full-time visa status in UNH graduate programs?\n\n"
            "User: international student health insurance?\n"
            "Rewrite: Is health insurance required for international graduate students at UNH?\n\n"
        )
    elif intent == "program_biotech":
        examples = (
            "Examples:\n"
            "User: biotech ms total credits?\n"
            "Rewrite: How many total credits are required for the UNH Biotechnology: Industrial and Biomedical Sciences (M.S) program?\n\n"
            "User: biotech core credits?\n"
            "Rewrite: How many core credits are required for the UNH Biotechnology: Industrial and Biomedical Sciences (M.S) program?\n\n"
        )
    elif intent == "program_cyber":
        examples = (
            "Examples:\n"
            "User: cybersecurity ms location?\n"
            "Rewrite: Where is the UNH Cybersecurity Engineering (M.S) program offered?\n\n"
            "User: cybersecurity learning modes?\n"
            "Rewrite: What learning modes are available in the UNH Cybersecurity Engineering (M.S) program?\n\n"
        )
    elif intent == "program_it":
        examples = (
            "Examples:\n"
            "User: it ms internship required?\n"
            "Rewrite: Is an internship required for the UNH Information Technology (M.S) program?\n\n"
            "User: it ms core credits?\n"
            "Rewrite: How many core credits are required for the UNH Information Technology (M.S) program?\n\n"
        )
    elif intent == "program_info":
        examples = (
            "Examples:\n"
            "User: what is grad 900?\n"
            "Rewrite: What is GRAD 900 in the UNH graduate catalog?\n\n"
            "User: independent study credits?\n"
            "Rewrite: How many independent study credits can count toward a UNH graduate degree?\n\n"
        )
    else:
        # general fallback
        examples = (
            "Examples:\n"
            "User: grad gpa for graduation?\n"
            "Rewrite: What GPA do UNH graduate students need in order to graduate?\n\n"
            "User: add drop deadline grad?\n"
            "Rewrite: What is the add/drop deadline for UNH graduate courses?\n\n"
        )

    return base + examples + f"User: {query}\nRewrite:"


def _candidate_score(original: str, rewritten: str) -> float:
    """Simple similarity score based on content token overlap."""
    orig = _content_tokens(original)
    rew = _content_tokens(rewritten)
    if not orig or not rew:
        return 0.0
    inter = len(orig & rew)
    return inter / max(1, len(orig))


def _cosine_similarity(v1, v2) -> float:
    """Cosine similarity between two embedding vectors represented as lists."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    num = 0.0
    sum1 = 0.0
    sum2 = 0.0
    for a, b in zip(v1, v2):
        num += a * b
        sum1 += a * a
        sum2 += b * b
    if sum1 == 0.0 or sum2 == 0.0:
        return 0.0
    return num / ((sum1 ** 0.5) * (sum2 ** 0.5))

def llm_rewrite(query: str) -> str:
    """Use the existing model to produce a precise, catalog-specific rewrite.
    Assumes call_model(prompt: str) -> str.
    """
    try:
        if _CALL_FN is None:
            if getattr(settings, "ENABLE_QUERY_REWRITER", False):
                # Surface a clearer message if rewrite was explicitly enabled but we couldn't import a model callable
                raise ImportError(f"[QueryRewrite] Could not locate a callable in models/ml_models.py. Original import error: {_IMPORT_ERROR}")
            # If rewriter disabled, we won't call a model anyway.
            return query

        intent = infer_intent(query)
        prompt = build_prompt(query, intent)

        best = None
        best_score = -1.0
        num_candidates = getattr(settings, "REWRITE_NUM_CANDIDATES", 3)

        # Optionally enable semantic similarity as an extra guardrail
        use_sim = bool(getattr(settings, "REWRITE_USE_SEMANTIC_SIMILARITY", False) and get_text_embedding)
        orig_emb = None
        if use_sim:
            try:
                orig_emb = get_text_embedding(query)  # type: ignore[operator]
            except Exception as e:
                print(f"[QueryRewrite] embedding for original failed, disabling sim check: {e}")
                use_sim = False

        for _ in range(max(1, num_candidates)):
            resp = _CALL_FN(prompt)
            rew = (resp or "").strip()
            low = rew.lower()

            # forbidden phrase checks
            # Never allow the model to echo instruction text
            if any(p in low for p in INSTRUCTION_PHRASES):
                continue
            # Block transfer boilerplate when the original question is not about transfer
            if any(p in low for p in TRANSFER_FORBIDDEN_PHRASES) and "transfer" not in query.lower():
                continue
            # Block invented negative eligibility statements if the original did not mention them
            if "are not eligible" in low and "not eligible" not in query.lower():
                continue

            # duplicate word check (e.g., "policy policy") -> clean up instead of rejecting
            if _has_dup_words(rew):
                rew = _dedupe_adjacent_words(rew)
                low = rew.lower()

            # basic content-token preservation: ensure we keep at least one content word from the original
            orig_tokens = _content_tokens(query)
            rew_tokens = _content_tokens(rew)
            if orig_tokens and not (orig_tokens & rew_tokens):
                continue

            # bidirectional policy noun checks: do not drop original policy terms or invent new ones
            orig_policy = _policy_terms(query)
            rew_policy = _policy_terms(rew)
            if orig_policy and not (orig_policy & rew_policy):
                # lost an important policy term
                continue
            if rew_policy - orig_policy:
                # introduced new policy terms that weren't asked about
                continue

            # length constraint using configurable bounds
            min_words = getattr(settings, "REWRITE_MIN_WORDS", 5)
            max_words = getattr(settings, "REWRITE_MAX_WORDS", 25)
            wcount = len(re.findall(r"[a-zA-Z]+", rew))
            if not (min_words <= wcount <= max_words):
                continue

            # Semantic sanity check: reject noisy/irrelevant rewrites
            if not _semantic_sanity(query, rew):
                continue

            # Enforce question form: rewrites should be questions, not declarative answers
            if not rew.strip().endswith("?"):
                continue

            # Optional semantic similarity guard: ensure the rewrite stays close in meaning
            if use_sim and orig_emb is not None:
                try:
                    rew_emb = get_text_embedding(rew)  # type: ignore[operator]
                    sim = _cosine_similarity(orig_emb, rew_emb)
                    min_sim = getattr(settings, "REWRITE_MIN_SIMILARITY", 0.6)
                    if sim < min_sim:
                        continue
                except Exception as e:
                    # If embedding computation fails, skip the sim check for this candidate
                    print(f"[QueryRewrite] embedding for candidate failed, skipping sim check: {e}")

            # If we reach here, candidate is valid; score it
            score = _candidate_score(query, rew)
            if score > best_score:
                best_score = score
                best = rew

        return best if best is not None else query
    except Exception as e:
        print(f"[QueryRewrite] call_model failed: {e}")
        return query

def transform_query(user_query: str) -> str:
    """
    Main entry: normalize -> domain rules -> optional LLM rewrite -> validate -> pick best.
    """
    original = user_query

    # Special-case overrides for a few known short queries
    key = original.strip().lower()
    if key in SPECIAL_REWRITES:
        return SPECIAL_REWRITES[key]

    cleaned  = normalize_text(original)
    expanded = apply_domain_rules(cleaned)
    expanded = _dedupe_adjacent_words(expanded)

    if getattr(settings, "ENABLE_QUERY_REWRITER", False):
        rewritten = llm_rewrite(expanded)
        if validate_rewrite(original, rewritten):
            return rewritten

    return expanded