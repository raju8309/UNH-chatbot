import csv
import json
import os
import pickle
import re
import threading
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, quote_plus

import numpy as np
import yaml
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import uvicorn

from dashboard import router as dashboard_router
from hierarchy import compute_tier
from text_fragments import build_text_fragment_url, choose_snippet, is_synthetic_label

# -----------------------
# Paths & Globals
# -----------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "scraper"
CACHE_PATH = DATA_DIR / "chunks_cache.pkl"
CHAT_LOG_PATH = "chat_logs.csv"

UNKNOWN = "I don't have that information."

CFG: Dict[str, Any] = {}
POLICY_TERMS: Tuple[str, ...] = ()

chunks_embeddings = None
chunk_texts: List[str] = []
chunk_sources: List[Dict[str, Any]] = []
chunk_meta: List[Dict[str, Any]] = []
CHUNK_NORMS: Optional[np.ndarray] = None

_LOG_LOCK = threading.Lock()
_APP_CONFIGURED = False  # guard so we don't add middleware twice

# -----------------------
# Session store 
# -----------------------
SESSIONS: Dict[str, Dict[str, Any]] = {}

def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def get_session(session_id: str) -> Dict[str, Any]:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {
            "intent": None,               
            "program_level": "unknown",    
            "program_alias": None,         
            "course_code": None,           
            "last_question": None,
            "last_answer": None,
            "last_retrieval_path": None,
            "history": [],
            "updated_at": _now_iso(),
        }
    return SESSIONS[session_id]

def update_session(session_id: str, **fields: Any) -> None:
    sess = get_session(session_id)
    sess.update(fields)
    sess["updated_at"] = _now_iso()

# -----------------------
# Session history helper
# -----------------------
def push_history(session_id: str, entry: Dict[str, Any]) -> None:
    """Append a history entry to the session, capped at 5 most recent turns."""
    sess = get_session(session_id)
    hist = sess.get("history")
    if not isinstance(hist, list):
        hist = []
    hist.append(entry)
    # Cap to last 5 items
    if len(hist) > 5:
        hist = hist[-5:]
    sess["history"] = hist
    sess["updated_at"] = _now_iso()

# -----------------------
# Models
# -----------------------
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load trained model, fallback on default
trained = Path(__file__).parent / "models" / "flan-t5-small-finetuned"
if trained.exists() and (trained / "config.json").exists():
    model = str(trained)
else:
    model = "google/flan-t5-small"
qa_pipeline = pipeline(
    "text2text-generation",
    model=model,
    device=-1,
)


# FastAPI app initialization
app = FastAPI()


# Startup event: auto-load config, data, mount frontend if run with uvicorn backend.main:app
@app.on_event("startup")
async def startup_event():
    """
    Automatically load configuration, catalog data, and mount frontend
    when running via 'uvicorn backend.main:app'.
    """
    try:
        load_retrieval_cfg()
        ensure_chat_log_file()
        load_initial_data()
        configure_app(app)
        print("[startup] Configuration, catalog, and frontend loaded successfully.")
    except Exception as e:
        print(f"[startup] WARNING: initialization failed: {e}")


def load_retrieval_cfg() -> None:
    """
    Loads retrieval configuration from config/retrieval.yaml.
    """
    global CFG, POLICY_TERMS
    cfg_path = Path(__file__).resolve().parent / "config" / "retrieval.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            CFG = yaml.safe_load(f) or {}
    else:
        CFG = {}
    # Fallbacks for missing keys, but all retrieval settings should be in YAML.
    if "policy_terms" not in CFG:
        CFG["policy_terms"] = []
    if "tier_boosts" not in CFG:
        CFG["tier_boosts"] = {1: 1.35, 2: 1.10, 3: 1.0, 4: 1.0}
    if "intent" not in CFG:
        CFG["intent"] = {
            "course_keywords": [],
            "degree_keywords": [],
            "course_code_regex": r"\b[A-Z]{3,5}\s?\d{3}\b",
        }
    if "nudges" not in CFG:
        CFG["nudges"] = {"policy_acadreg_url": 1.15}
    if "guarantees" not in CFG:
        CFG["guarantees"] = {
            "ensure_tier1_on_policy": True,
            "ensure_tier4_on_program": True
        }
    if "tier4_gate" not in CFG:
        CFG["tier4_gate"] = {"use_embedding": True, "min_title_sim": 0.42, "min_alt_sim": 0.38}
    # Insert search pool defaults if not present
    if "search" not in CFG:
        CFG["search"] = {"topn_base": 40, "topn_with_alias": 80}
    else:
        # keep sensible fallbacks if keys are missing
        CFG["search"]["topn_base"] = CFG["search"].get("topn_base", 40)
        CFG["search"]["topn_with_alias"] = CFG["search"].get("topn_with_alias", 80)
    # Diversity / de-dup defaults (config-gated)
    if "diversity" not in CFG:
        CFG["diversity"] = {}
    CFG["diversity"]["enable"] = CFG["diversity"].get("enable", True)
    CFG["diversity"]["same_url_penalty"] = CFG["diversity"].get("same_url_penalty", 0.9)
    CFG["diversity"]["same_block_drop"] = CFG["diversity"].get("same_block_drop", True)
    POLICY_TERMS = tuple(CFG.get("policy_terms", []))


def _compute_meta_from_source(src: Dict[str, Any]) -> Dict[str, Any]:
    return compute_tier(src.get("url", ""), src.get("title", ""))

def save_chunks_cache() -> None:
    if not chunk_texts or chunks_embeddings is None:
        return
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(
            {
                "texts": chunk_texts,
                "sources": chunk_sources,
                "embeddings": chunks_embeddings,
                "meta": chunk_meta,
                "norms": CHUNK_NORMS,
            },
            f,
        )
    print(f"saved {len(chunk_texts)} chunks to cache")

def load_chunks_cache() -> bool:
    global chunk_texts, chunk_sources, chunks_embeddings, chunk_meta
    global CHUNK_NORMS
    if not CACHE_PATH.exists():
        return False
    with open(CACHE_PATH, "rb") as f:
        data = pickle.load(f)
    chunk_texts = data.get("texts", [])
    chunk_sources = data.get("sources", [])
    chunks_embeddings = data.get("embeddings")
    CHUNK_NORMS = data.get("norms")
    cached_meta = data.get("meta")
    if cached_meta:
        chunk_meta = cached_meta
    else:
        chunk_meta = [_compute_meta_from_source(src) for src in chunk_sources]
    if chunks_embeddings is None or not chunk_texts:
        chunk_meta = []
        chunk_sources = []
        chunk_texts = []
        return False
    if CHUNK_NORMS is None and chunks_embeddings is not None:
        CHUNK_NORMS = np.linalg.norm(chunks_embeddings, axis=1)
    print(f"Loaded {len(chunk_texts)} chunks from cache.")
    return True

def load_json_file(path: str) -> None:
    global chunks_embeddings, chunk_texts, chunk_sources, chunk_meta

    page_count = 0

    def _process_section(section: Dict[str, Any], parent_title: str = "", base_url: str = "") -> None:
        nonlocal page_count  # Access the outer variable
        """Recursively process sections and subsections"""
        if not isinstance(section, dict):
            return
            
        page_count += 1
        
        title = section.get("title", "").strip()
        url = section.get("page_url", base_url).strip()
        
        # Build hierarchical title
        full_title = f"{parent_title} - {title}" if parent_title and title else (title or parent_title)
        
        # Process text content - add everything no matter length
        texts = section.get("text", [])
        if texts:
            for text_item in texts:
                if isinstance(text_item, str) and text_item.strip():
                    add_chunk(text_item.strip(), full_title, url)
        
        # Process lists - only keep content more than 5 words to filter out navigational lists
        lists = section.get("lists", [])
        if lists:
            for list_group in lists:
                if isinstance(list_group, list):
                    for item in list_group:
                        if isinstance(item, str) and item.strip():
                            if (len(item) > 5):
                                add_chunk(item.strip(), full_title, url)
        
        # Process subsections recursively
        subsections = section.get("subsections", [])
        if subsections:
            for subsection in subsections:
                _process_section(subsection, full_title, url)

    def add_chunk(text: str, title: str, url: str) -> None:
        new_texts.append(text)
        src = {"title": title, "url": url}
        new_sources.append(src)
        new_meta.append(_compute_meta_from_source(src))

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"ERROR: Could not parse JSON from {path}")
        return

    new_texts: List[str] = []
    new_sources: List[Dict[str, Any]] = []
    new_meta: List[Dict[str, Any]] = []

    # Process the root structure
    if isinstance(data, dict):
        pages = data.get("pages", [])
        if isinstance(pages, list):
            for page in pages:
                if isinstance(page, dict):
                    page_title = page.get("page_title", "")
                    page_url = page.get("page_url", "")
                    sections = page.get("sections", [])
                    if isinstance(sections, list):
                        for section in sections:
                            _process_section(section, page_title, page_url)
    
    if new_texts:
        new_embeds = embed_model.encode(new_texts, convert_to_numpy=True)
        if chunks_embeddings is None:
            chunks_embeddings = new_embeds
        else:
            chunks_embeddings = np.vstack([chunks_embeddings, new_embeds])
        chunk_texts.extend(new_texts)
        chunk_sources.extend(new_sources)
        chunk_meta.extend(new_meta)
        global CHUNK_NORMS
        if chunks_embeddings is not None:
            CHUNK_NORMS = np.linalg.norm(chunks_embeddings, axis=1)
        print(f"Loaded {len(new_texts)} chunks from {path}")
    else:
        print(f"WARNING: no text found in {path}")
    print(f"[loader] Pages parsed: {page_count}")
    print(f"[loader] New chunks: {len(new_texts)}  |  Total chunks: {len(chunk_texts)}")

def load_catalog(path: Path) -> None:
    if path.exists():
        load_json_file(str(path))
    else:
        print(f"WARNING: {path} not found, skipping.")


def _program_intent(query: str) -> bool:
    q = (query or "")
    ql = q.lower()
    intent = CFG.get("intent", {})
    course_kw = intent.get("course_keywords", [])
    degree_kw = intent.get("degree_keywords", [])
    code_rx = intent.get("course_code_regex", r"\b[A-Z]{3,5}\s?\d{3}\b")
    course_code = re.search(code_rx, q)
    return any(k in ql for k in (course_kw + degree_kw)) or bool(course_code)

def _tier_boost(tier: int) -> float:
    return float(CFG.get("tier_boosts", {}).get(tier, 1.0))

def _is_acad_reg_url(url: str) -> bool:
    return isinstance(url, str) and "/graduate/academic-regulations-degree-requirements/" in url


# --- Diversity helper: drop near-duplicate chunks from the same URL/section
# Keeps order; if we drop items, we top-up from `ordered` to keep K items.
# A "duplicate" is defined by (base URL without fragment, normalized title, first 80 chars of normalized text).
# This is deliberately conservative so it will not impact distinct passages on different sections.

def _dedup_same_block_keep_order(final: List[int], k: int, ordered: List[int]) -> List[int]:
    try:
        if not CFG.get("diversity", {}).get("same_block_drop", True):
            return final[:k]
        seen: set = set()
        out: List[int] = []
        for i in final:
            if i >= len(chunk_sources) or i >= len(chunk_texts):
                continue
            src = chunk_sources[i] or {}
            base_url = (src.get("url") or "").split("#")[0]
            title_key = (src.get("title") or "").strip().lower()
            text_key = re.sub(r"\s+", " ", (chunk_texts[i] or "").strip().lower())[:160]
            key = (base_url, title_key, text_key[:80])
            if key in seen:
                continue
            seen.add(key)
            out.append(i)
        # Top-up from the ordered candidate list if we removed too many
        if len(out) < k:
            for i in ordered:
                if len(out) >= k:
                    break
                if i in out:
                    continue
                if i >= len(chunk_sources) or i >= len(chunk_texts):
                    continue
                src = chunk_sources[i] or {}
                base_url = (src.get("url") or "").split("#")[0]
                title_key = (src.get("title") or "").strip().lower()
                text_key = re.sub(r"\s+", " ", (chunk_texts[i] or "").strip().lower())[:160]
                key = (base_url, title_key, text_key[:80])
                if key in seen:
                    continue
                seen.add(key)
                out.append(i)
        return out[:k]
    except Exception:
        # Fail open if anything unexpected happens
        return final[:k]

def _title_for_sim(src: Dict[str, Any]) -> str:
    title = (src.get("title") or "").strip()
    url = (src.get("url") or "")
    path = urlparse(url).path if url else ""
    segs = [s for s in path.split("/") if s]
    tail = " ".join(segs[-2:]) if segs else ""
    return (title + " " + tail).strip()

# --- Policy term detector
def _has_policy_terms(text: str) -> bool:
    t = (text or "").lower()
    return any(kw in t for kw in [
        "gpa", "good standing", "probation", "dismissal", "grade", "b-", "b minus", "c grade", "c-", "c minus", "minimum gpa"
    ])

# --- Admissions term detector and URL helpers ---
def _has_admissions_terms(text: str) -> bool:
    t = (text or "").lower()
    return any(kw in t for kw in [
        "admission", "admissions", "apply", "application", "requirements", "gre", "gmat",
        "test score", "test scores", "english proficiency", "toefl", "ielts",
        "letters of recommendation", "recommendation letter"
    ])


def _is_admissions_url(url: str) -> bool:
    return isinstance(url, str) and "/graduate/general-information/admissions/" in url


def _is_degree_requirements_url(url: str) -> bool:
    return isinstance(url, str) and "/graduate/academic-regulations-degree-requirements/degree-requirements/" in url

def _tier4_is_relevant_embed(query: str, idx: int) -> bool:
    gate = CFG.get("tier4_gate", {})
    if not gate.get("use_embedding", True):
        return True
    if idx >= len(chunk_sources):
        return False
    cand = _title_for_sim(chunk_sources[idx])
    if not cand:
        return False
    q_vec, c_vec = embed_model.encode([query, cand], convert_to_numpy=True)
    sim = float(np.dot(q_vec, c_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(c_vec) + 1e-8))
    thresh = float(gate.get("min_title_sim", 0.42))
    return sim >= thresh

def _same_program_family(u1: str, u2: str) -> bool:
    """
    Consider two URLs the 'same program family' if the tuple
    ('programs-study', <school>, <program>) matches.

    Example:
      /graduate/programs-study/computing/information-technology-ms/...
      /graduate/programs-study/computing/information-technology-ms/#overviewtext
    """
    def key(u: str):
        try:
            parts = [s for s in urlparse(u or "").path.split("/") if s]  # drop empties
            if "programs-study" not in parts:
                return ()
            i = parts.index("programs-study")
            # Need: programs-study, <school>, <program>
            core = parts[i : i + 3 + 1]  # ['programs-study', school, program]
            if len(core) < 3:
                return ()
            # return exactly ('programs-study', school, program)
            return tuple(core[:3])
        except Exception:
            return ()
    k1, k2 = key(u1), key(u2)
    return k1 != () and (k1 == k2)

# --- Degree flags and alias/level conflict utilities ---
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


_LEVEL_HINT_TOKEN = {"phd": "ph.d.", "grad": "master's", "certificate": "certificate", "undergrad": "bachelor"}


def _alias_conflicts_with_level(alias: Optional[Dict[str, str]], level: str) -> bool:
    if not alias or not level or level == "unknown":
        return False
    is_ms, is_phd, is_cert, is_ug = _degree_flags_from_title_url(alias.get("title", ""), alias.get("url", ""))
    if level == "certificate":
        return not is_cert
    if level == "phd":
        return not is_phd
    if level == "grad":
        # treat grad as MS-ish; allow PhD too
        return not (is_ms or is_phd)
    if level == "undergrad":
        return not is_ug
    return False

# ---------- Course helpers  ----------
COURSE_CODE_RX = re.compile(r"\b([A-Z]{2,5})\s*-?\s*(\d{3}[A-Z]?)\b")
COURSE_LINE_CREDITS_RX = re.compile(r"\bCredits:\s*([0-9]+)\b", re.I)
COURSE_LINE_PREREQ_RX = re.compile(r"\bPrerequisite\(s\):\s*(.+?)(?:\.\s*$|$)", re.I)
COURSE_LINE_GRADEMODE_RX = re.compile(r"\bGrade\s*Mode:\s*([A-Za-z/ \-]+)$", re.I)

def _detect_course_code(message: str) -> Optional[Dict[str, str]]:
    """
    Detects a course code in the message and returns {'subj','num','norm'}
    Accepts: 'MATH954', 'MATH-954', 'MATH 954'
    """
    if not message:
        return None
    m = COURSE_CODE_RX.search(message.upper())
    if not m:
        return None
    subj = re.sub(r"[^A-Z]", "", m.group(1).upper())
    num = m.group(2).upper().replace("-", "").replace(" ", "")
    # split trailing letter from number, keep as one token
    return {"subj": subj, "num": num, "norm": f"{subj} {num}"}

def _course_search_url(norm: str) -> str:
    # Catalog uses /search/?P=SUBJ%20NUM
    return f"https://catalog.unh.edu/search/?P={quote_plus(norm)}"

def _url_contains_course(url: str, norm: str) -> bool:
    # match either encoded or raw space
    if not url:
        return False
    u = url.lower()
    raw = norm.lower().replace(" ", "%20")
    return (f"/search/?p={norm.lower().replace(' ', '%20')}" in u) or (f"/search/?p={norm.lower().replace(' ', '+')}" in u) or (f"/search/?p={norm.lower()}" in u)

def _title_starts_with_course(title: str, norm: str) -> bool:
    t = (title or "").upper().strip()
    return t.startswith(norm.upper())

def _extract_course_fallbacks(chunks: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, Optional[str]]:
    """
    Look for 'Credits:', 'Prerequisite(s):', 'Grade Mode:' in top course chunks.
    Returns dict with any found fields.
    """
    out = {"credits": None, "prereqs": None, "grademode": None}
    for text, _ in chunks:
        if not text:
            continue
        if out["credits"] is None:
            m = COURSE_LINE_CREDITS_RX.search(text)
            if m:
                out["credits"] = m.group(1)
        if out["prereqs"] is None:
            m = COURSE_LINE_PREREQ_RX.search(text)
            if m:
                out["prereqs"] = m.group(1).strip()
        if out["grademode"] is None:
            m = COURSE_LINE_GRADEMODE_RX.search(text)
            if m:
                out["grademode"] = m.group(1).strip()
    return out

def search_chunks(
    query: str,
    topn: int = 40,
    k: int = 5,
    alias_url: Optional[str] = None,
    intent_key: Optional[str] = None,
    course_norm: Optional[str] = None,
):
    if chunks_embeddings is None or not chunk_texts:
        return [], []

    q_vec = embed_model.encode([query], convert_to_numpy=True)[0]
    # Compute cosine similarities using cached norms
    global CHUNK_NORMS
    if CHUNK_NORMS is None:
        CHUNK_NORMS = np.linalg.norm(chunks_embeddings, axis=1)
    chunk_norms = CHUNK_NORMS
    query_norm = np.linalg.norm(q_vec)
    # Avoid division by zero
    valid_chunks = chunk_norms > 1e-8
    sims = np.zeros(len(chunks_embeddings))
    if query_norm > 1e-8:
        sims[valid_chunks] = (chunks_embeddings[valid_chunks] @ q_vec) / (chunk_norms[valid_chunks] * query_norm)
    
    # Get top candidates
    cand_idxs = np.argsort(-sims)[:topn * 2].tolist()
    
    # Enhanced filtering based on query analysis
    q_lower = (query or "").lower()
    allow_program = bool(alias_url)
    looks_policy = any(term in q_lower for term in POLICY_TERMS)
    looks_admissions = (intent_key == "admissions") or any(tok in q_lower for tok in [
        "admission", "admissions", "apply", "gre", "gmat", "test score", "test scores", "toefl", "ielts"
    ])
    # Extract key terms from query for better matching
    query_terms = set(re.findall(r'\b\w+\b', q_lower))
    
    filtered: List[int] = []
    for i in cand_idxs:
        if i >= len(chunk_texts):
            continue
        chunk_text_lower = chunk_texts[i].lower()
        meta_i = chunk_meta[i] if i < len(chunk_meta) else {}
        tier = meta_i.get("tier", 2)
        # Policy hygiene: tighten what we allow when the query looks like a policy question
        if looks_policy:
            # Drop Tier-3 course-description chunks unless they include policy terms
            if tier == 3:
                if not _has_policy_terms(chunk_text_lower):
                    continue
            # Tier-4 is only allowed if it's the SAME program family AND the chunk actually has policy terms.
            if tier == 4:
                src_i = chunk_sources[i] if i < len(chunk_sources) else {}
                url_i = (src_i.get("url") or "")
                if alias_url:
                    if not _same_program_family(url_i, alias_url):
                        continue
                    if not _has_policy_terms(chunk_text_lower):
                        continue
                else:
                    # No specific program was requested → keep policy answers general (Tier-1/2)
                    continue
        if looks_admissions:
            # Keep Tier-3 only if it actually contains admissions terms
            if tier == 3 and not _has_admissions_terms(chunk_text_lower):
                continue
            # Tier-4 allowed; we boost the right ones during rescoring
        # More lenient relevance check
        term_matches = len(query_terms.intersection(set(re.findall(r'\b\w+\b', chunk_text_lower))))
        if term_matches == 0 and sims[i] < 0.1:  # Lower threshold
            continue
        # If the user asked about a specific course, Tier-2 must match that course code
        if course_norm and tier == 2:
            src_i = chunk_sources[i] if i < len(chunk_sources) else {}
            title_i = (src_i.get("title") or "")
            url_i = (src_i.get("url") or "")
            if not (_url_contains_course(url_i, course_norm) or _title_starts_with_course(title_i, course_norm)):
                continue
        # Apply tier filtering
        if tier in (3, 4) and not allow_program:
            continue
        if tier == 4 and allow_program:
            src_i = chunk_sources[i] if i < len(chunk_sources) else {}
            if alias_url and _same_program_family(src_i.get("url", ""), alias_url):
                pass  # keep it
            else:
                if not _tier4_is_relevant_embed(query, i):
                    continue
        filtered.append(i)

    if not filtered:
        allowed_tiers = {1, 2} if not allow_program else {1, 2, 3, 4}
        filtered = [
            i for i in range(len(chunk_meta))
            if ((chunk_meta[i] or {}).get("tier") in allowed_tiers)
        ] or list(range(len(chunk_meta)))

    policy_nudge = float(CFG.get("nudges", {}).get("policy_acadreg_url", 1.15))
    rescored = []
    for i in filtered:
        meta_i = chunk_meta[i] if i < len(chunk_meta) else {}
        src_i = chunk_sources[i] if i < len(chunk_sources) else {}
        tier = meta_i.get("tier", 2)
        base = float(sims[i]) * _tier_boost(tier)

        # Nudge for academic regulations when the user asks a policy-ish thing
        nudge = policy_nudge if looks_policy and _is_acad_reg_url(src_i.get("url", "")) else 1.0

        # Stronger bias for same-program when alias is set (follow-ups)
        same_prog_bonus = 1.0
        if alias_url and _same_program_family(src_i.get("url", ""), alias_url):
            same_prog_bonus = 1.9

        # Course bias: if a course code is present, strongly prefer Tier-2 search page / course-titled chunks
        course_bonus = 1.0
        if course_norm:
            url = (src_i.get("url") or "")
            title = (src_i.get("title") or "")
            if _url_contains_course(url, course_norm):
                course_bonus = 1.8
            elif _title_starts_with_course(title, course_norm):
                course_bonus = 1.4
            elif tier in (3, 4):
                course_bonus = 0.9
            elif tier == 1 and looks_policy:
                course_bonus = 1.0

        # Admissions-specific nudges
        admissions_bonus = 1.0
        if looks_admissions:
            url_i = (src_i.get("url") or "")
            txt_i = (chunk_texts[i] or "").lower()
            if _is_admissions_url(url_i):
                admissions_bonus *= 1.6  # strong boost for the university admissions page
            if _has_admissions_terms(txt_i):
                admissions_bonus *= 1.25  # prefer chunks that actually talk admissions or GRE
            if _is_degree_requirements_url(url_i):
                admissions_bonus *= 0.85  # demote generic degree requirements for admissions questions

        rescored.append((i, base * nudge * same_prog_bonus * course_bonus * admissions_bonus))

    rescored.sort(key=lambda x: x[1], reverse=True)
    ordered = [i for i, _ in rescored]

    # For policy queries, try to lead with a Tier-1 result
    if looks_policy:
        lead_t1 = None
        for i in ordered:
            if (chunk_meta[i] or {}).get("tier") == 1:
                lead_t1 = i
                break
        if lead_t1 is not None and ordered and ordered[0] != lead_t1:
            ordered.remove(lead_t1)
            ordered.insert(0, lead_t1)

    # Build preliminary top-k
    final: List[int] = []
    if looks_policy and bool(CFG.get("guarantees", {}).get("ensure_tier1_on_policy", True)):
        has_tier1 = any((chunk_meta[i] or {}).get("tier") == 1 for i in ordered[:k])
        if not has_tier1:
            best_t1_idx = -1
            best_t1_score = -1.0
            for i in range(len(chunk_meta)):
                meta_i = chunk_meta[i] or {}
                if meta_i.get("tier") == 1:
                    sc = float(sims[i]) * _tier_boost(1) * policy_nudge
                    if sc > best_t1_score:
                        best_t1_score = sc
                        best_t1_idx = i
            if best_t1_idx != -1:
                final.append(best_t1_idx)

    for i in ordered:
        if len(final) >= k:
            break
        if i not in final:
            final.append(i)

    final = final[:k]

    #  when user has a program alias, ensure at least one Tier-4 from the "same program"
    want_program_page = bool(alias_url) and bool(CFG.get("guarantees", {}).get("ensure_tier4_on_program", True))
    if want_program_page:
        def _is_t4(i: int) -> bool:
            return (chunk_meta[i] or {}).get("tier") == 4
        
        def _same_family_idx(i: int) -> bool:
            try:
                return _same_program_family((chunk_sources[i] or {}).get("url", ""), alias_url or "")
            except Exception:
                return False

        has_tier4_same = any(_is_t4(i) and _same_family_idx(i) for i in final)

        if not has_tier4_same:
            # Pick best same-family Tier-4 if available; else best any Tier-4 as fallback (from rescored)
            best_same = (-1, -1.0)
            best_any = (-1, -1.0)
            for i, score in rescored:
                if not _is_t4(i):
                    continue
                if _same_family_idx(i):
                    if score > best_same[1]:
                        best_same = (i, score)
                if score > best_any[1]:
                    best_any = (i, score)

            # --------- widen search to ALL chunks if no same-program Tier-4 was in rescored ----------
            if best_same[0] == -1:
                for j in range(len(chunk_meta)):
                    meta_j = chunk_meta[j] or {}
                    if meta_j.get("tier") != 4:
                        continue
                    if not _same_family_idx(j):
                        continue
                    # approximate score compatible with above: similarity * tier boost
                    sc = float(sims[j]) * _tier_boost(4)
                    if sc > best_same[1]:
                        best_same = (j, sc)
            # ----------------------------------------------------------------------------------------------

            inject = best_same[0] if best_same[0] != -1 else best_any[0]
            if inject != -1 and inject not in final:
                final.append(inject)
                # de-dup, preserve order, then trim to k with preference for the injected + first item
                seen_idx = set()
                dedup = []
                for ii in final:
                    if ii not in seen_idx:
                        seen_idx.add(ii)
                        dedup.append(ii)
                if len(dedup) > k:
                    keepers = {dedup[0], inject}
                    trimmed = [x for x in dedup if x in keepers]
                    for x in dedup:
                        if len(trimmed) >= k:
                            break
                        if x not in trimmed:
                            trimmed.append(x)
                    dedup = trimmed[:k]
                final = dedup

    # If a course is asked, try to ensure at least one course page is included
    if course_norm and not any(_url_contains_course((chunk_sources[i] or {}).get("url", ""), course_norm) for i in final):
        # find best course page among rescored
        best_course = (-1, -1.0)
        for i, score in rescored:
            if _url_contains_course((chunk_sources[i] or {}).get("url", ""), course_norm) or \
               _title_starts_with_course((chunk_sources[i] or {}).get("title", ""), course_norm):
                if score > best_course[1]:
                    best_course = (i, score)
        if best_course[0] != -1:
            final[-1] = best_course[0]

    # Final, conservative de-dup of near-identical chunks from the same page/section
    if CFG.get("diversity", {}).get("enable", True):
        final = _dedup_same_block_keep_order(final, k, ordered)
    # Build retrieval_path
    retrieval_path = []
    for rank, i in enumerate(final, start=1):
        src = chunk_sources[i] if i < len(chunk_sources) else {}
        meta = chunk_meta[i] if i < len(chunk_meta) else {}
        retrieval_path.append(
            {
                "rank": rank,
                "idx": i,
                "score": round(float(sims[i]), 6),
                "title": src.get("title"),
                "url": src.get("url"),
                "tier": meta.get("tier"),
                "tier_name": meta.get("tier_name"),
                "text": chunk_texts[i] if i < len(chunk_texts) else ""
            }
        )
    return final, retrieval_path

def _wrap_sources_with_text_fragments(sources_with_passages, question: str):
    wrapped = []
    for passage, src in sources_with_passages:
        url = src.get("url", "")
        if not url or is_synthetic_label(passage):
            wrapped.append({**src, "url": url})
            continue
        snippet = choose_snippet(passage, hint=question, max_chars=160)
        wrapped.append({**src, "url": build_text_fragment_url(url, text=snippet) if snippet else url})
    return wrapped

# --------  intent & program detectors --------
_INTENT_KEYWORDS = {
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
    "program_options": [
        "thesis", "non-thesis", "project option", "project", "exam option", "comprehensive exam"
    ],
}

_INTENT_TEMPLATES = {
    "gpa_minimum": "minimum GPA to stay in good standing",
    "admissions": "admission requirements",
    "credit_transfer": "transfer credit policy",
    "registration": "add/drop and withdrawal deadlines",
    "course_info": "course details and prerequisites",
    "degree_credits": "total credits required",
    "program_options": "thesis vs project/exam options",
}

_LEVEL_HINTS = {
    "undergrad": ["undergrad", "bachelor", "bs", "ba"],
    "grad": ["graduate", "grad", "master", "ms", "m.s.", "ma", "m.a."],
    "phd": ["phd", "ph.d.", "doctoral", "doctorate"],
    "certificate": ["certificate", "grad certificate", "graduate certificate"],
}

# --- Helper for explicit program mention ---
def _explicit_program_mention(text: str) -> bool:
    t = (text or "").lower()
    # Heuristic: contains a degree word and a scoping preposition like "in"/"for" or the word "program"
    has_degree_word = bool(re.search(r"\b(ms|m\.s\.|master'?s|phd|ph\.d\.|doctoral|doctorate|certificate)\b", t))
    has_scope_word = (" program " in f" {t} ") or bool(re.search(r"\b(in|for)\b", t))
    return has_degree_word and has_scope_word

# simple in-memory index of program pages for fuzzy matching
_PROGRAM_PAGES: List[Dict[str, str]] = []  # {"title", "url", "norm"}

_SECTION_STOPWORDS = tuple([
    "upon completion", "program learning outcomes", "admission requirements",
    "requirements", "application requirements", "core courses", "electives",
    "overview", "policies", "sample", "plan of study"
])


# follow-up detector (short, context-carrying messages)
_FOLLOWUP_HINTS = ("for ", "now ", "do it for", "for the", "make that for",
                   "do that for", "do it", "that", "this", "same")

def _looks_like_followup(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    # very short or starts with common follow-up stems
    return (len(t) <= 60 and any(t.startswith(h) for h in _FOLLOWUP_HINTS)) or t in ("same", "that", "this")
# ---------------------------------------------------------

def _norm(s: Optional[str]) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _looks_like_program_url(u: str) -> bool:
    return (
        "/graduate/programs-study/" in u
        and "/search/?" not in u
        and "/academic-regulations-degree-requirements/" not in u
    )

def _build_program_index() -> None:
    _PROGRAM_PAGES.clear()
    for src, meta in zip(chunk_sources, chunk_meta):
        try:
            tier = (meta or {}).get("tier")
            if tier not in (3, 4):
                continue
            title = (src.get("title") or "").strip()
            url = src.get("url") or ""
            if not title or not url:
                continue
            if not _looks_like_program_url(url):
                continue
            norm_title = _norm(title)
            if any(s in norm_title for s in _SECTION_STOPWORDS):
                continue
            _PROGRAM_PAGES.append({"title": title, "url": url, "norm": norm_title})
        except Exception:
            continue

def _detect_intent(message: str, prev_intent: Optional[str]) -> Optional[str]:
    q = _norm(message)
    #  sticky intent on likely follow-ups
    if _looks_like_followup(message) and prev_intent:
        return prev_intent

    best = prev_intent
    best_hits = 0
    for intent, kws in _INTENT_KEYWORDS.items():
        hits = sum(1 for kw in kws if kw in q)
        if hits > best_hits:
            best_hits = hits
            best = intent
    # If any course code appears, intent is course_info
    if _detect_course_code(message):
        best = "course_info"
    return best

# --- Helper: auto intent from topic, for follow-up/topic reuse ---
def _auto_intent_from_topic(message: str) -> Optional[str]:
    """Heuristically infer a policy intent when none was detected.
    This makes follow-ups like "do it for MS" reuse the prior topic with a stable intent.
    """
    q = _norm(message)
    if not q:
        return None
    # GPA / standing / grading
    if any(tok in q for tok in ["good standing", "probation", "dismissal", "minimum gpa", "gpa", "grade", "c grade", "b-", "b minus", "c-", "c minus"]):
        return "gpa_minimum"
    # Registration policies
    if any(tok in q for tok in ["withdraw", "withdrawal", "drop", "add drop", "deadline", "last day to"]):
        return "registration"
    # Degree credit total
    if any(tok in q for tok in ["credits required", "how many credits", "total credits", "credit requirement", "credit hours"]):
        return "degree_credits"
    # Program options (thesis vs project/exam)
    if any(tok in q for tok in ["thesis", "non thesis", "non-thesis", "project", "exam option", "comprehensive exam"]):
        return "program_options"
    # Course info
    if _detect_course_code(message):
        return "course_info"
    # Admissions (GRE/GMAT etc.)
    if any(tok in q for tok in ["gre", "gmat", "toefl", "ielts", "admission", "admissions", "apply", "recommendation"]):
        return "admissions"
    return None


def _detect_program_level(message: str, fallback: str) -> str:
    q = _norm(message)
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
    for level, hints in _LEVEL_HINTS.items():
        if any(h in q for h in hints):
            return level
    return fallback or "unknown"

# --- Correction/negation detection for program level ---
def _detect_correction_or_negation(text: str) -> Dict[str, Optional[str]]:
    """
    Detect correction or negation phrases indicating a change or removal of program level.
    Returns dict like {"negated_level": str|None, "new_level": str|None}.
    """
    t = (text or "").lower()
    result = {"negated_level": None, "new_level": None}
    # Negations
    for lvl, hints in _LEVEL_HINTS.items():
        for h in hints:
            if f"not {h}" in t or re.search(rf"not in (a|the)?\s*{h}", t):
                result["negated_level"] = lvl
    # Affirmations / declarations
    for lvl, hints in _LEVEL_HINTS.items():
        for h in hints:
            if re.search(rf"\bfor (a|the)?\s*{h}\b", t) or re.search(rf"i'?m in (a|the)?\s*{h}\b", t):
                result["new_level"] = lvl
    # Fallback: explicit phrases like "actually, ...", "do it for grad"
    if "actually" in t or "instead" in t:
        for lvl, hints in _LEVEL_HINTS.items():
            if any(h in t for h in hints):
                result["new_level"] = lvl
    return result

def _match_program_alias(message: str) -> Optional[Dict[str, str]]:
    if not _PROGRAM_PAGES:
        return None

    q_raw = (message or "").strip()
    q_norm = _norm(q_raw)

    # ---- Degree / award hints from the user's message ----
    t = f" {q_raw.lower()} "
    wants_ms = bool(re.search(r"\bms\b|\bm\.s\.?\b|\bmaster'?s\b", t))
    wants_phd = bool(re.search(r"\bphd\b|\bph\.d\.?\b|\bdoctoral\b|\bdoctorate\b", t))
    wants_cert = "certificate" in t
    wants_undergrad = bool(re.search(r"\bundergrad\b|\bbachelor'?s\b|\bbs\b|\bba\b", t))

    def _degree_flags(rec: Dict[str, str]) -> Tuple[bool, bool, bool, bool]:
        title = (rec.get("title") or "").lower()
        url = (rec.get("url") or "").lower()
        is_cert = ("certificate" in title) or ("certificate" in url)
        is_phd = ("phd" in title) or ("ph.d" in title) or ("/phd" in url)
        # Common MS indicators in title/URL
        is_ms = (
            "(m.s" in title or " m.s" in title or " m.s." in title or " ms" in title or
            "/ms" in url or "-ms/" in url or url.endswith("-ms/") or "-ms#" in url
        )
        is_undergrad = ("b.s" in title) or ("b.a" in title) or ("/bs" in url) or ("/ba" in url)
        return is_ms, is_phd, is_cert, is_undergrad

    def _degree_allowed(rec: Dict[str, str]) -> bool:
        is_ms, is_phd, is_cert, is_undergrad = _degree_flags(rec)
        # If the user specified a degree/award, filter accordingly
        if wants_ms:
            if is_cert or is_phd:
                return False
        if wants_phd:
            if is_cert or is_ms:
                return False
        if wants_cert:
            if not is_cert:
                return False
        if wants_undergrad:
            if not is_undergrad:
                return False
        return True

    # Candidate pool respecting degree hints when present
    if wants_ms or wants_phd or wants_cert or wants_undergrad:
        CANDS_ALL = [rec for rec in _PROGRAM_PAGES if _degree_allowed(rec)]
        if not CANDS_ALL:
            # If filtering removed everything, fall back to all programs
            CANDS_ALL = list(_PROGRAM_PAGES)
    else:
        CANDS_ALL = list(_PROGRAM_PAGES)

    # ---- Fast path A: title containment (normalized tokens) ----
    for rec in CANDS_ALL:
        tnorm = rec.get("norm") or ""
        if not tnorm:
            continue
        if q_norm and (q_norm in tnorm or tnorm in q_norm):
            return {"title": rec["title"], "url": rec["url"]}

    # ---- Fast path B: slug containment in URL ----
    def _slugify(s: str) -> str:
        s = re.sub(r"[^a-z0-9\s]", " ", (s or "").lower())
        s = re.sub(r"\s+", "-", s).strip("-")
        return s

    DROP = {"ms", "m.s", "m.s.", "program", "degree", "graduate", "master", "masters"}
    q_tokens = [tok for tok in q_norm.split() if tok and tok not in DROP]
    q_slug = _slugify(" ".join(q_tokens))  # e.g., "information-technology"
    if q_slug:
        for rec in CANDS_ALL:
            url = rec.get("url") or ""
            if q_slug in url:
                return {"title": rec["title"], "url": rec["url"]}

    # ---- Embedding shortlist (token-overlap) then semantic pick ----
    q_tokens_set = set(q_norm.split())
    cands_overlap = [rec for rec in CANDS_ALL if (set((rec.get("norm") or "").split()) & q_tokens_set)]
    if not cands_overlap:
        cands_overlap = CANDS_ALL  # last resort: consider all within degree-filtered set

    titles = [rec["title"] for rec in cands_overlap]
    vecs = embed_model.encode([q_raw] + titles, convert_to_numpy=True)
    qv, tv = vecs[0], vecs[1:]
    sims = (tv @ qv) / (np.linalg.norm(tv, axis=1) * np.linalg.norm(qv) + 1e-8)
    best_idx = int(np.argmax(sims))
    best = cands_overlap[best_idx]

    # If the user explicitly asked for MS/PhD/Certificate, apply a slight preference boost
    # to matches whose degree flags align; otherwise, fall back to threshold as before.
    if wants_ms or wants_phd or wants_cert or wants_undergrad:
        is_ms, is_phd, is_cert, is_ug = _degree_flags(best)
        aligns = (
            (wants_ms and is_ms) or (wants_phd and is_phd) or (wants_cert and is_cert) or (wants_undergrad and is_ug)
        )
        if not aligns:
            # Try to find the top aligned candidate, if any
            aligned_idx = None
            best_aligned_score = -1.0
            for i, rec in enumerate(cands_overlap):
                ims, iphd, icert, iug = _degree_flags(rec)
                if (wants_ms and ims) or (wants_phd and iphd) or (wants_cert and icert) or (wants_undergrad and iug):
                    if sims[i] > best_aligned_score:
                        best_aligned_score = sims[i]
                        aligned_idx = i
            if aligned_idx is not None:
                best = cands_overlap[aligned_idx]
                best_idx = aligned_idx

    if float(sims[best_idx]) < 0.45:
        return None
    return {"title": best["title"], "url": best["url"]}

# -----------------------
# Regex fallbacks (policy + course)
# -----------------------
_CREDIT_LINE_RX = re.compile(
    r"(?:(?:minimum|at least|a total(?: of)?|total(?: of)?)\s+)?(\d{1,3})\s*(?:credit|credits|cr)\b",
    re.IGNORECASE,
)

def _extract_best_credits(chunks: List[Tuple[str, Dict[str, Any]]]) -> Optional[Tuple[str, Dict[str, Any], str]]:
    """
    Look through chunk texts for an explicit credit count.
    Prefer lines containing 'minimum'/'total'/'required'.
    Returns (passage_text, source_dict, normalized_answer_string) or None.
    """
    best: Optional[Tuple[str, Dict[str, Any], str, int]] = None  # (text, src, ans, weight)
    for text, src in chunks:
        for m in _CREDIT_LINE_RX.finditer(text or ""):
            num = m.group(1)
            # Heuristic weight
            span_text = text[max(0, m.start()-60): m.end()+60]
            w = 1
            if re.search(r"\bminimum\b|\brequired\b|\btotal\b", span_text, re.I):
                w += 2
            # prefer plausible ranges
            try:
                n = int(num)
                if 6 <= n <= 90:
                    w += 1
            except Exception:
                pass
            ans = f"{num}"
            cand = (text, src, ans, w)
            if best is None or cand[3] > best[3]:
                best = cand
    if best:
        return (best[0], best[1], best[2])
    return None

def _extract_gre_requirement(question: str, chunks: List[Tuple[str, Dict[str, Any]]]) -> Optional[Tuple[str, Dict[str, Any], str]]:
    """
    If the user asked about GRE/GMAT, try to synthesize a yes/no.
    """
    qn = _norm(question)
    if not any(tok in qn for tok in ["gre", "g r e", "gmat", "g m a t", "test score", "test scores"]):
        return None

    for text, src in chunks:
        t = (text or "").lower()
        if "gre" in t or "gmat" in t or "test score" in t or "test scores" in t:
            if re.search(r"\bnot required\b|\bno gre\b|\bwaived\b|\bno (?:gmat|gre) required\b", t):
                return (text, src, "No")
            if re.search(r"\brequired\b|\bmust submit\b|\bofficial scores\b", t):
                return (text, src, "Yes")
    return None

# QA core (now alias- & course-aware)
def build_context_from_indices(idxs: List[int]) -> Tuple[List[Tuple[str, Dict[str, Any]]], str]:
    """
    Build (top_chunks, context_string) from precomputed indices without re-searching.
    """
    if not idxs:
        return [], ""
    top_chunks = [(chunk_texts[i], chunk_sources[i]) for i in idxs if i < len(chunk_texts)]
    parts = []
    for text, source in top_chunks:
        title = source.get("title", "Source")
        title = title.split(" - ")[-1] if " - " in title else title
        parts.append(f"{title}: {text}")
    return top_chunks, "\n\n".join(parts)


def _answer_question(
    question: str,
    alias_url: Optional[str] = None,
    intent_key: Optional[str] = None,
    course_norm: Optional[str] = None,
):
    # widen initial pool when we have a program alias, so the “same program” pages make the candidate set
    topn_cfg = CFG.get("search", {})
    topn_local = int(topn_cfg.get("topn_with_alias", 80)) if alias_url else int(topn_cfg.get("topn_base", 40))
    idxs, retrieval_path = search_chunks(
        question, topn=topn_local, k=5, alias_url=alias_url, intent_key=intent_key, course_norm=course_norm
    )
    # Build context from the *same* indices to keep generation/sources aligned
    top_chunks, context = build_context_from_indices(idxs)

    # prefer a same-program, "facty" chunk at the top for generation
    def _url_same_family(u: str, base: Optional[str]) -> bool:
        return bool(base) and _same_program_family(u or "", base or "")

    def _has_direct_fact(text: str) -> bool:
        t = (text or "")[:500]
        return bool(re.search(r"\b\d{1,3}\b", t)) or bool(re.search(r"\b(credit|credits|gpa|gre|thesis|option)\b", t, re.I))

    if alias_url and idxs:
        top5 = idxs[:5]
        prefer_idx = None
        for i in top5:
            src = (chunk_sources[i] if i < len(chunk_sources) else {}) or {}
            if _url_same_family(src.get("url",""), alias_url) and _has_direct_fact(chunk_texts[i]):
                prefer_idx = i
                break
        if prefer_idx is not None and prefer_idx != idxs[0]:
            # move preferred index to the front for generation context
            idxs.remove(prefer_idx)
            idxs.insert(0, prefer_idx)
            # also re-rank retrieval_path for transparency
            for r in retrieval_path:
                if r.get("idx") == prefer_idx:
                    r["rank"] = 1
            rank_counter = 2
            for r in retrieval_path:
                if r.get("idx") != prefer_idx:
                    r["rank"] = rank_counter
                    rank_counter += 1
            # Rebuild generation context to match the new order
            top_chunks, context = build_context_from_indices(idxs)

    if not idxs:
        return "I couldn't find relevant information in the catalog.", [], retrieval_path

    # Compose a course-aware hint if applicable
    course_hint = ""
    if course_norm and intent_key == "course_info":
        course_hint = f" Focus on credits, prerequisites, and grade mode for {course_norm} if present."

    # Run the generator
    try:
        long_result = qa_pipeline(
            get_prompt(question + course_hint, context),
            max_new_tokens=128,
            repetition_penalty=1.2,
            no_repeat_ngram_size=2,
        )
        answer = long_result[0]["generated_text"].strip()
    except Exception as exc:
        return f"ERROR running local model: {exc}", [], retrieval_path

    # Regex fallbacks if model punts
    def _looks_idk(a: str) -> bool:
        return bool(re.search(r"\bi don'?t know\b", (a or "").lower()))

    enriched_sources = _wrap_sources_with_text_fragments(top_chunks, question)

    # --- Policy guardrail: deterministic grading fact when appropriate ---
    def _tier1_grading_present(paths: List[Dict[str, Any]]) -> bool:
        for r in paths or []:
            title = (r.get("title") or "").lower()
            url = (r.get("url") or "").lower()
            if "/graduate/academic-regulations-degree-requirements/grading/" in url or "academic standards" in title:
                return True
        return False

    is_grade_topic = bool(re.search(r"\b(gpa|grade|good standing|probation|dismissal|b-\b|c grade|c-\b)\b", (question or "").lower())) or (intent_key == "gpa_minimum")
    if is_grade_topic and _tier1_grading_present(retrieval_path):
        answer = ("Graduate credit is only granted for courses completed with B- or higher. "
                  "Individual programs may set stricter requirements; see your program page for details.")

    # degree_credits fallback
    if intent_key == "degree_credits":
        hit = _extract_best_credits(top_chunks)
        if hit:
            _, src, num = hit
            idk = _looks_idk(answer)
            if idk or not re.search(r"\b\d{1,3}\b", answer):
                answer = f"{num}."

    # GRE fallback (when the question mentions GRE/GMAT)
    asked_gre = bool(re.search(r"\b(gre|gmat|test score|test scores)\b", (question or "").lower()))
    gre_hit = _extract_gre_requirement(question, top_chunks)
    if gre_hit:
        _, _, yesno = gre_hit
        if _looks_idk(answer):
            answer = f"{yesno}."
    elif asked_gre and (intent_key == "admissions"):
        # Admissions guardrail when no explicit GRE text is retrieved
        answer = ("GRE requirements are program-specific. Many UNH graduate programs do not require GRE, "
                  "but some do. Check the Admission Requirements section on your program page for the current policy.")

    # Course fallbacks
    if course_norm and intent_key == "course_info":
        cf = _extract_course_fallbacks(top_chunks)
        need_help = _looks_idk(answer) or (not re.search(r"credits|prereq|grade", answer, re.I))
        if need_help and any(cf.values()):
            parts = []
            if cf["credits"]:
                parts.append(f"Credits: {cf['credits']}")
            if cf["prereqs"]:
                parts.append(f"Prerequisite(s): {cf['prereqs']}")
            if cf["grademode"]:
                parts.append(f"Grade Mode: {cf['grademode']}")
            if parts:
                answer = ". ".join(parts) + "."

    # Build citations (use all 5 retrieved, with text fragments from top 3)
    enriched_all = enriched_sources + [src for _, src in top_chunks[3:]]

    seen = set()
    citation_lines = []
    for src in enriched_all:
        key = (src.get("title"), src.get("url"))
        if key in seen:
            continue
        seen.add(key)
        line = f"- {src.get('title', 'Source')}"
        if src.get("url"):
            line += f" ({src['url']})"
        citation_lines.append(line)


    return answer, citation_lines, retrieval_path


# ----------------------- FOLLOW-UP ANSWER FORMATTER -----------------------
def _format_followup_answer(answer: str, sess: Dict[str, Any], intent_key: Optional[str], is_followup: bool) -> str:
    """Prefix answers with 'For <Program/Course>, ...' when a label exists; avoid duplicates and keep casing/punctuation natural."""
    text = (answer or "").strip()
    if not text or text.lower() == UNKNOWN.lower():
        return answer or ""

    # Choose display label (prefer course code, else program title)
    label = None
    course = sess.get("course_code") if isinstance(sess, dict) else None
    if isinstance(course, dict) and course.get("norm"):
        label = course["norm"]
    alias = sess.get("program_alias") if isinstance(sess, dict) else None
    if label is None and isinstance(alias, dict) and alias.get("title"):
        label = alias["title"]
    if not label:
        return answer

    # Normalize duplicate program titles like "X - X" -> "X"
    try:
        parts = [p.strip() for p in re.split(r"\s*-\s*", label) if p.strip()]
        if len(parts) == 2 and parts[0].lower() == parts[1].lower():
            label = parts[0]
    except Exception:
        pass

    body = text.lstrip()

    # If it already begins with the same label, do nothing
    if body.lower().startswith(f"for {label.lower()}"):
        return answer

    # If it begins with a different "For ..., " prefix, leave as-is (avoid churn)
    if body.lower().startswith("for "):
        try:
            head = body[4:].split(",", 1)[0].strip().lower()
            if head == label.lower():
                return answer
        except Exception:
            pass

    # Lowercase the first letter after the comma when the sentence looks like Title case (avoid proper noun issues)
    first = body[:1]
    rest = body[1:]
    if first.isupper() and (rest[:1].islower() if rest else False) and not body.startswith("I "):
        body = first.lower() + rest

    return f"For {label}, {body}"


def get_prompt(question, context):
    return (
        "Using ONLY the provided context, write a concise explanation in exactly 2–3 complete sentences.\n"
        "Mention requirements, deadlines, or procedures if they are present.\n"
        f"If the context is insufficient, output exactly: {UNKNOWN}\n"
        "Do not include assumptions, examples, or general knowledge beyond the context.\n\n"
        f"Context: {context}\n\n"
        f"Question: {question}\n\n"
        f"Detailed explanation:"
    )

# cache key includes alias_url + intent + course_norm to avoid mixing scoped answers
@lru_cache(maxsize=128)
def _cached_answer_core(cache_key: str):
    msg, alias, intent, course = cache_key.split("|||", 3)
    alias = alias or None
    intent = intent or None
    course = course or None
    return _answer_question(msg, alias_url=alias, intent_key=intent, course_norm=course)

def cached_answer_with_path(message: str, alias_url: Optional[str] = None, intent_key: Optional[str] = None, course_norm: Optional[str] = None):
    cache_key = f"{message}|||{alias_url or ''}|||{intent_key or ''}|||{course_norm or ''}"
    return _cached_answer_core(cache_key)

# -----------------------
# Small util
# -----------------------
def base_doc_id(url: str) -> str:
    if not url:
        return "catalog"
    p = urlparse(url)
    name = (Path(p.path).name or "").rstrip("/")
    if not name and p.path:
        name = Path(p.path).parts[-1]
    slug = (name or "catalog").replace(".html", "").replace(".htm", "") or "catalog"
    return slug

# -----------------------
# API models
# -----------------------
class ChatRequest(BaseModel):
    message: str
    history: Optional[List[str]] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    retrieval_path: List[Dict[str, Any]]

# -----------------------
# Debug endpoints
# -----------------------
@app.get("/debug/tier-counts")
def tier_counts():
    counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for meta in chunk_meta:
        tier = (meta or {}).get("tier")
        if tier in counts:
            counts[tier] += 1
    return {
        "tier1": counts[1],
        "tier2": counts[2],
        "tier3": counts[3],
        "tier4": counts[4],
        "total": len(chunk_meta),
    }

# -------- Session utilities endpoints  --------
@app.get("/_session/{session_id}")
def get_session_debug(session_id: str):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
    return SESSIONS[session_id]

@app.post("/reset-session")
def reset_one_session(x_session_id: Optional[str] = Header(default=None)):
    if not x_session_id:
        raise HTTPException(status_code=400, detail="Please include X-Session-Id header")
    if x_session_id in SESSIONS:
        del SESSIONS[x_session_id]
        return {"status": "session_cleared", "session_id": x_session_id}
    return {"status": "no_session_to_clear", "session_id": x_session_id}

@app.post("/reset")
async def reset_chat():
    SESSIONS.clear()
    return {"status": "cleared"}


# Logging

def log_chat_to_csv(question: str, answer: str, sources: List[str]) -> None:
    ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    row = [ts, question, answer, json.dumps(sources, ensure_ascii=False)]
    with _LOG_LOCK:
        with open(CHAT_LOG_PATH, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)


@app.post("/chat", response_model=ChatResponse)
async def answer_question(request: ChatRequest, x_session_id: Optional[str] = Header(default=None)):
    if not x_session_id:
        raise HTTPException(status_code=400, detail="Please include X-Session-Id header")

    sess = get_session(x_session_id)
    incoming_message = request.message if not isinstance(request.message, list) else " ".join(request.message)

    #  update session context ---
    new_intent = _detect_intent(incoming_message, prev_intent=sess.get("intent"))
    # --- strong per-message intent override to avoid sticky bleed-through ---
    _msg_l = (incoming_message or "").lower()
    if any(tok in _msg_l for tok in ["gpa", "good standing", "probation", "dismissal", "minimum gpa", "c grade", "b-", "c-"]):
        new_intent = "gpa_minimum"
    elif any(tok in _msg_l for tok in ["add drop", "add/drop", "withdraw", "withdrawal", "last day to drop", "registration"]):
        new_intent = "registration"
    new_level = _detect_program_level(incoming_message, fallback=sess.get("program_level") or "unknown")
    match = _match_program_alias(incoming_message)
    new_alias = match or sess.get("program_alias")

    # Correction / negation handling
    corr = _detect_correction_or_negation(incoming_message)
    if corr.get("negated_level"):
        neg = corr["negated_level"]
        if sess.get("program_level") == neg:
            new_level = "unknown"
            new_alias = None
    if corr.get("new_level"):
        new_level = corr["new_level"]
    # If correction detected, keep previous intent/topic (reuse last question context)
    if any(corr.values()):
        if sess.get("last_question"):
            # mark as follow-up reuse
            incoming_message = sess.get("last_question")
            print(f"[correction] Reusing previous topic due to correction: {incoming_message}")

    # Degree-aware alias realignment when level is set and current alias conflicts
    try:
        if new_alias and isinstance(new_alias, dict) and new_level and new_level != "unknown":
            if _alias_conflicts_with_level(new_alias, new_level):
                level_hint = _LEVEL_HINT_TOKEN.get(new_level, "")
                hinted_message = incoming_message + (f" {level_hint}" if level_hint else "")
                rematch = _match_program_alias(hinted_message)
                new_alias = rematch if rematch else None
    except Exception:
        pass

    # Determine if this turn is a follow-up (or a correction/affirmation)
    is_followup = _looks_like_followup(request.message) or any(corr.values())
    base_topic = incoming_message
    if is_followup and sess.get("last_question"):
        base_topic = sess.get("last_question")

    # Preserve prior program on short follow-ups unless the user clearly names a new one
    try:
        explicit_prog = _explicit_program_mention(request.message)
        if is_followup and sess.get("program_alias") and not explicit_prog:
            new_alias = sess.get("program_alias")
        elif match:
            # If user explicitly named a program, accept the new match
            if explicit_prog:
                new_alias = match
            else:
                # No explicit mention: prefer sticking with prior alias if present
                new_alias = sess.get("program_alias") or match
        else:
            new_alias = sess.get("program_alias")
    except Exception:
        pass

    # If intent was not confidently detected, infer from the base topic
    if not new_intent:
        inferred = _auto_intent_from_topic(base_topic)
        if inferred:
            new_intent = inferred
        elif sess.get("intent"):
            # sticky previous intent for short follow-ups
            new_intent = sess.get("intent")

    # Secondary course detection fallback for explicit follow-up like: "what about DATA 800?"
    detected_course = _detect_course_code(base_topic)
    if not detected_course and (sess.get("intent") == "course_info"):
        if re.search(r"\bwhat about\b", incoming_message, re.I) or COURSE_CODE_RX.search(incoming_message.upper()):
            detected_course = _detect_course_code(incoming_message)

    # Save session updates
    update_session(
        x_session_id,
        intent=new_intent,
        program_level=new_level,
        program_alias=new_alias,
        course_code=detected_course,
        last_question=base_topic,
    )

    # Compose a scoped message using the base topic and any new scope
    scoped_message = base_topic
    alias_url = None
    if new_alias and isinstance(new_alias, dict):
        alias_url = new_alias.get("url")

    intent_key = new_intent or sess.get("intent")

    # Prefer course-specific scoping when the intent is course_info
    detected_course_obj = detected_course
    if detected_course_obj and (intent_key == "course_info"):
        scoped_message = f"course details and prerequisites for {detected_course_obj['norm']}"
    else:
        # Build from intent templates if available
        if intent_key in _INTENT_TEMPLATES:
            scoped_message = _INTENT_TEMPLATES[intent_key]
            if new_alias and isinstance(new_alias, dict):
                scoped_message += f" for {new_alias['title']}"
            # Add level parenthetical when known and helpful
            if new_level and new_level != "unknown" and intent_key != "degree_credits":
                scoped_message += f" ({new_level})"
        else:
            # No template -> keep the base topic but append scope if present
            if new_alias and isinstance(new_alias, dict):
                scoped_message = f"{base_topic} for {new_alias['title']}"
            if new_level and new_level != "unknown" and (not new_alias or intent_key not in _INTENT_TEMPLATES):
                scoped_message += f" ({new_level})"

    # Retrieve / Answer (alias- & course-aware, with guarantees/biases)
    course_norm = detected_course_obj["norm"] if detected_course_obj else None
    if intent_key == "gpa_minimum":
        alias_url = None  # keep GPA answers general (Tier-1/2)
    answer, sources, retrieval_path = cached_answer_with_path(
        scoped_message, alias_url=alias_url, intent_key=intent_key, course_norm=course_norm
    )
    # Post-process: format follow-up answers with program/course context when appropriate
    try:
        sess_obj = get_session(x_session_id)
        answer = _format_followup_answer(answer, sess_obj, intent_key, True)
    except Exception:
        pass

    update_session(
        x_session_id,
        last_answer=answer,
        last_retrieval_path=retrieval_path,
    )

    # Record this turn in bounded session history (last 5)
    try:
        push_history(
            x_session_id,
            {
                "timestamp": _now_iso(),
                "question": base_topic,
                "scoped_message": scoped_message,
                "answer": answer,
                "intent": intent_key,
                "program_level": new_level,
                "program_alias": (new_alias or {}).get("title") if isinstance(new_alias, dict) else None,
                "course_code": (detected_course_obj or {}).get("norm") if isinstance(detected_course_obj, dict) else None,
                "retrieval_path": retrieval_path,
            },
        )
    except Exception:
        # Non-fatal: history is a debug aid
        pass

    log_chat_to_csv(base_topic, answer, sources)
    return ChatResponse(answer=answer, sources=sources, retrieval_path=retrieval_path)


def configure_app(app_instance: FastAPI) -> None:
    app_instance.include_router(dashboard_router)

    PUBLIC_URL = os.getenv("PUBLIC_URL", "http://localhost:8003/")
    app_instance.add_middleware(
        CORSMiddleware,
        allow_origins=[PUBLIC_URL],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    frontend_path = BASE_DIR / "frontend" / "out"
    if frontend_path.is_dir():
        app_instance.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")
        print("Mounted frontend from:", frontend_path)

def ensure_chat_log_file() -> None:
    if not os.path.isfile(CHAT_LOG_PATH):
        with open(CHAT_LOG_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["timestamp", "question", "answer", "sources_json"])

def load_initial_data() -> None:
    loaded_from_cache = load_chunks_cache()
    if not loaded_from_cache:
        filenames = ["unh_catalog.json"]
        for name in filenames:
            load_catalog(DATA_DIR / name)
        save_chunks_cache()
    _build_program_index()  # program title/url index for aliasing


if __name__ == "__main__":
    load_retrieval_cfg()
    ensure_chat_log_file()
    load_initial_data()
    configure_app(app)
    uvicorn.run(app, host="0.0.0.0", port=8003, reload=False)
