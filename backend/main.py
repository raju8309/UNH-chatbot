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

CFG: Dict[str, Any] = {}
POLICY_TERMS: Tuple[str, ...] = ()

chunks_embeddings = None
chunk_texts: List[str] = []
chunk_sources: List[Dict[str, Any]] = []
chunk_meta: List[Dict[str, Any]] = []

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
            "updated_at": _now_iso(),
        }
    return SESSIONS[session_id]

def update_session(session_id: str, **fields: Any) -> None:
    sess = get_session(session_id)
    sess.update(fields)
    sess["updated_at"] = _now_iso()

# -----------------------
# Models
# -----------------------
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
qa_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    device=-1,
)

app = FastAPI()


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
            },
            f,
        )
    print(f"saved {len(chunk_texts)} chunks to cache")

def load_chunks_cache() -> bool:
    global chunk_texts, chunk_sources, chunks_embeddings, chunk_meta
    if not CACHE_PATH.exists():
        return False
    with open(CACHE_PATH, "rb") as f:
        data = pickle.load(f)
    chunk_texts = data.get("texts", [])
    chunk_sources = data.get("sources", [])
    chunks_embeddings = data.get("embeddings")
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
    print(f"Loaded {len(chunk_texts)} chunks from cache.")
    return True

def load_json_file(path: str) -> None:
    global chunks_embeddings, chunk_texts, chunk_sources, chunk_meta

    def _iter_pages(obj: Any):
        if isinstance(obj, list):
            for rec in obj:
                if isinstance(rec, dict):
                    yield rec.get("page_title") or rec.get("title"), rec.get("page_url", ""), rec.get("sections", [])
            return
        if isinstance(obj, dict):
            pages = obj.get("pages")
            if isinstance(pages, list):
                for rec in pages:
                    if isinstance(rec, dict):
                        yield rec.get("page_title") or rec.get("title"), rec.get("page_url", ""), rec.get("sections", [])
                return
            if "sections" in obj:
                yield obj.get("page_title") or obj.get("title"), obj.get("page_url", ""), obj.get("sections", [])
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(rec, dict):
                        yield rec.get("page_title") or rec.get("title"), rec.get("page_url", ""), rec.get("sections", [])
        except Exception:
            return

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        data = None

    if data is not None:
        page_iter = _iter_pages(data)
    else:
        page_iter = _iter_pages("")

    new_texts: List[str] = []
    new_sources: List[Dict[str, Any]] = []
    new_meta: List[Dict[str, Any]] = []

    def add_piece(text_str: str, title_str: str, url_str: str) -> None:
        if not text_str:
            return
        new_texts.append(text_str)
        src = {"title": title_str, "url": url_str}
        new_sources.append(src)
        new_meta.append(_compute_meta_from_source(src))

    page_count = 0
    for page_title, page_url, sections in page_iter:
        page_count += 1
        if not isinstance(sections, list):
            sections = [sections] if isinstance(sections, dict) else []
        for sub in sections:
            if not isinstance(sub, dict):
                continue
            sec_title = sub.get("title", "")
            full_title = f"{page_title} – {sec_title}" if sec_title else (page_title or sec_title)
            sec_url = sub.get("page_url", "") or page_url
            for p in sub.get("text", []) or []:
                if isinstance(p, str):
                    add_piece(p, full_title, sec_url)
            for li in sub.get("lists", []) or []:
                if isinstance(li, list):
                    for item in li:
                        if isinstance(item, str):
                            add_piece(item, full_title, sec_url)
            for link in sub.get("links", []) or []:
                if isinstance(link, dict):
                    label = link.get("label")
                    link_url = link.get("url")
                    if label and link_url:
                        add_piece(f"Courses: {label}", label, link_url)

    if new_texts:
        new_embeds = embed_model.encode(new_texts, convert_to_numpy=True)
        if chunks_embeddings is None:
            chunks_embeddings = new_embeds
        else:
            chunks_embeddings = np.vstack([chunks_embeddings, new_embeds])
        chunk_texts.extend(new_texts)
        chunk_sources.extend(new_sources)
        chunk_meta.extend(new_meta)
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

def _title_for_sim(src: Dict[str, Any]) -> str:
    title = (src.get("title") or "").strip()
    url = (src.get("url") or "")
    path = urlparse(url).path if url else ""
    segs = [s for s in path.split("/") if s]
    tail = " ".join(segs[-2:]) if segs else ""
    return (title + " " + tail).strip()

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

def _search_chunks(
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
    denom = (np.linalg.norm(chunks_embeddings, axis=1) * np.linalg.norm(q_vec)) + 1e-8
    sims = (chunks_embeddings @ q_vec) / denom

    cand_idxs = np.argsort(-sims)[:topn].tolist()
    q_lower = (query or "").lower()
    allow_program = _program_intent(query) or bool(alias_url)
    looks_policy = any(term in q_lower for term in POLICY_TERMS)

    filtered: List[int] = []
    for i in cand_idxs:
        meta_i = chunk_meta[i] if i < len(chunk_meta) else {}
        tier = meta_i.get("tier", 2)
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

        rescored.append((i, base * nudge * same_prog_bonus * course_bonus))

    rescored.sort(key=lambda x: x[1], reverse=True)
    ordered = [i for i, _ in rescored]

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
            }
        )
    return final, retrieval_path

def get_top_chunks_policy(question: str, top_k: int = 5):
    idxs, _ = _search_chunks(question, topn=40, k=top_k)
    return [(chunk_texts[i], chunk_sources[i]) for i in idxs]

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

def _combine_answers(short_answer: str, long_answer: str) -> str:
    # Normalize
    s = (short_answer or "").strip()
    l = (long_answer or "").strip()

    def _is_idk(txt: str) -> bool:
        return bool(re.search(r"\bi don'?t know\b", (txt or "").lower()))

    # If one says I don't know and the other has content, drop the IDK
    if _is_idk(s) and l:
        s = ""
    if _is_idk(l) and s:
        l = ""

    s = s.strip(" \n\r\t.:")
    if s and l:
        if l.lower().startswith(s.lower()):
            combined = l
        else:
            combined = f"{s}. {l}"
    else:
        combined = s or l

    
    sentences = re.split(r"(?<=[.!?]) +", combined)
    return " ".join(sentences[:3]).strip()

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
}

_INTENT_TEMPLATES = {
    "gpa_minimum": "minimum GPA to stay in good standing",
    "admissions": "admission requirements",
    "credit_transfer": "transfer credit policy",
    "registration": "add/drop and withdrawal deadlines",
    "course_info": "course details and prerequisites",
    "degree_credits": "total credits required",
}

_LEVEL_HINTS = {
    "undergrad": ["undergrad", "bachelor", "bs", "ba"],
    "grad": ["graduate", "grad", "master", "ms", "m.s.", "ma", "m.a."],
    "phd": ["phd", "ph.d.", "doctoral", "doctorate"],
    "certificate": ["certificate", "grad certificate", "graduate certificate"],
}

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

def _match_program_alias(message: str) -> Optional[Dict[str, str]]:
    if not _PROGRAM_PAGES:
        return None

    q_raw = (message or "").strip()
    q_norm = _norm(q_raw)

    #  Fast path A: title containment (normalized tokens)
    
    for rec in _PROGRAM_PAGES:
        tnorm = rec["norm"]  # normalized title
        if not tnorm:
            continue
        if q_norm and (q_norm in tnorm or tnorm in q_norm):
            return {"title": rec["title"], "url": rec["url"]}

   
    # build a coarse slug from the question and try to find it in the URL
    def _slugify(s: str) -> str:
        s = re.sub(r"[^a-z0-9\s]", " ", (s or "").lower())
        s = re.sub(r"\s+", "-", s).strip("-")
        return s

    # drop generic words so "information technology" survives but "ms program" won't dominate
    DROP = {"ms", "m.s", "m.s.", "program", "degree", "graduate", "master", "masters"}
    q_tokens = [t for t in q_norm.split() if t and t not in DROP]
    q_slug = _slugify(" ".join(q_tokens))  # e.g., "information-technology"

    if q_slug:
        for rec in _PROGRAM_PAGES:
            url = rec.get("url") or ""
            if q_slug in url:
                return {"title": rec["title"], "url": rec["url"]}

    # ---- Existing logic (fallback): token-overlap shortlist - embedding pick
    q_tokens_set = set(q_norm.split())
    cands = [rec for rec in _PROGRAM_PAGES if (set(rec["norm"].split()) & q_tokens_set)]
    if not cands:
        cands = _PROGRAM_PAGES  # last resort: consider all

    titles = [rec["title"] for rec in cands]
    vecs = embed_model.encode([q_raw] + titles, convert_to_numpy=True)
    qv, tv = vecs[0], vecs[1:]
    sims = (tv @ qv) / (np.linalg.norm(tv, axis=1) * np.linalg.norm(qv) + 1e-8)
    best_idx = int(np.argmax(sims))
    best = cands[best_idx]
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
def _answer_question(
    question: str,
    alias_url: Optional[str] = None,
    intent_key: Optional[str] = None,
    course_norm: Optional[str] = None,
):
    # widen initial pool when we have a program alias, so the “same program” pages make the candidate set
    topn_local = 120 if alias_url else 40
    idxs, retrieval_path = _search_chunks(
        question, topn=topn_local, k=5, alias_url=alias_url, intent_key=intent_key, course_norm=course_norm
    )

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

    if not idxs:
        return "I couldn't find relevant information in the catalog.", [], retrieval_path

    # Keep retrieval list, but only feed top 3 chunks to the generator to reduce noise
    top_for_gen = [(chunk_texts[i], chunk_sources[i]) for i in idxs[:3]]

    # Compose a course-aware hint if applicable
    course_hint = ""
    if course_norm and intent_key == "course_info":
        course_hint = f" Focus on credits, prerequisites, and grade mode for {course_norm} if present."

    context = " ".join(text for text, _ in top_for_gen)

    # Run the generator
    try:
        short_prompt = (
            "Answer the question using ONLY the provided context. "
            "Start with a short, factual answer (number/date/Yes-No) if possible. "
            "If the context does not directly answer, reply with \"I don't know.\" "
            "Do not restate the question." + course_hint +
            "\n\nContext:\n" + context + f"\n\nQuestion: {question}\nAnswer:"
        )
        long_prompt = (
            "Provide a brief 2–3 sentence explanation using ONLY the provided context. "
            "Avoid tautologies or generic definitions. "
            "If the context lacks the info, say you don't know." + course_hint +
            "\n\nContext:\n" + context + f"\n\nQuestion: {question}\nAnswer:"
        )
        short_result = qa_pipeline(short_prompt, max_new_tokens=32)
        long_result = qa_pipeline(long_prompt, max_new_tokens=128)
        answer = _combine_answers(short_result[0]["generated_text"], long_result[0]["generated_text"])
    except Exception as exc:
        return f"ERROR running local model: {exc}", [], retrieval_path

    # Regex fallbacks if model punts
    def _looks_idk(a: str) -> bool:
        return bool(re.search(r"\bi don'?t know\b", (a or "").lower()))

    enriched_sources = _wrap_sources_with_text_fragments(top_for_gen, question)

    # degree_credits fallback
    if intent_key == "degree_credits":
        hit = _extract_best_credits(top_for_gen)
        if hit:
            _, src, num = hit
            idk = _looks_idk(answer)
            if idk or not re.search(r"\b\d{1,3}\b", answer):
                answer = f"{num}."

    # GRE fallback (when the question mentions GRE/GMAT)
    gre_hit = _extract_gre_requirement(question, top_for_gen)
    if gre_hit:
        _, _, yesno = gre_hit
        if _looks_idk(answer):
            answer = f"{yesno}."

    # Course fallbacks
    if course_norm and intent_key == "course_info":
        cf = _extract_course_fallbacks(top_for_gen)
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
    full_top = [(chunk_texts[i], chunk_sources[i]) for i in idxs]
    enriched_all = enriched_sources + [src for _, src in full_top[3:]]

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
    new_level = _detect_program_level(incoming_message, fallback=sess.get("program_level") or "unknown")
    match = _match_program_alias(incoming_message)
    new_alias = match or sess.get("program_alias")

    # detect course code (and follow-up handling) ---
    detected_course = _detect_course_code(incoming_message)

    # Follow-up pattern
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
        last_question=incoming_message,
    )

    # Compose a scoped question when we have an intent + program alias OR a course code
    scoped_message = incoming_message
    alias_url = None
    if new_alias and isinstance(new_alias, dict):
        alias_url = new_alias.get("url")

    intent_key = new_intent or sess.get("intent")

    # If we have a course, we prefer course-specific scoped prompt
    if detected_course and intent_key == "course_info":
        scoped_message = f"course details and prerequisites for {detected_course['norm']}"

    # Otherwise, program-scoped message as before
    elif new_alias and intent_key in _INTENT_TEMPLATES:
        prog_title = new_alias["title"]
        scoped_message = f"{_INTENT_TEMPLATES[intent_key]} for {prog_title}"
        if intent_key != "degree_credits" and new_level and new_level != "unknown":
            scoped_message += f" ({new_level})"

    # final safeguard to keep prior intent on short follow-ups
    if _looks_like_followup(incoming_message) and sess.get("intent"):
        intent_key = sess.get("intent")

    # Retrieve / Answer (alias- & course-aware, with guarantees/biases)
    course_norm = detected_course["norm"] if detected_course else None
    answer, sources, retrieval_path = cached_answer_with_path(
        scoped_message, alias_url=alias_url, intent_key=intent_key, course_norm=course_norm
    )

    update_session(
        x_session_id,
        last_answer=answer,
        last_retrieval_path=retrieval_path,
    )

    log_chat_to_csv(incoming_message, answer, sources)
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
