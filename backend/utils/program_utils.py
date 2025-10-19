import re
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
import numpy as np
from models.ml_models import get_embed_model
from config.settings import get_config

# in-memory index of program pages
_PROGRAM_PAGES: List[Dict[str, str]] = []

# section stopwords to filter out
_SECTION_STOPWORDS = (
    "upon completion", "program learning outcomes", "admission requirements",
    "requirements", "application requirements", "core courses", "electives",
    "overview", "policies", "sample", "plan of study"
)

def normalize_text(text: Optional[str]) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def looks_like_program_url(url: str) -> bool:
    return (
        "/graduate/programs-study/" in url
        and "/search/?" not in url
        and "/academic-regulations-degree-requirements/" not in url
    )

def build_program_index(chunk_sources: List[Dict], chunk_meta: List[Dict]) -> None:
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
            
            if not looks_like_program_url(url):
                continue
            
            norm_title = normalize_text(title)
            
            # skip section-specific pages
            if any(s in norm_title for s in _SECTION_STOPWORDS):
                continue
            
            _PROGRAM_PAGES.append({
                "title": title,
                "url": url,
                "norm": norm_title
            })
        except Exception:
            continue
    
    print(f"Built program index with {len(_PROGRAM_PAGES)} programs")

def is_blocked_program_url(url: str) -> bool:
    cfg = get_config()
    try:
        u = (url or "").lower()
        for slug in cfg.get("program_blocklist", []) or []:
            if slug and slug.lower() in u:
                return True
        return False
    except Exception:
        return False

def mentions_blocked_program(message: str) -> bool:
    cfg = get_config()
    try:
        q = (message or "").lower()
        for tok in cfg.get("program_blocklist_tokens", []) or []:
            if tok and tok.lower() in q:
                return True
        return False
    except Exception:
        return False


def match_program_alias(message: str) -> Optional[Dict[str, str]]:
    if not _PROGRAM_PAGES:
        return None

    cfg = get_config()
    q_raw = (message or "").strip()
    q_norm = normalize_text(q_raw)

    # --- Preferred alias routing (Biotechnology M.S.) ---
    try:
        pref_map = (cfg.get("preferred_program_aliases") or {})
        biotech_pref = pref_map.get("biotechnology_ms", "")
        if biotech_pref and re.search(r"\bbiotechnology\b", q_norm) and re.search(r"\b(m\.?s\.?|master'?s)\b", q_norm):
            # if the preferred program exists in the index, return it immediately
            for rec in _PROGRAM_PAGES:
                if biotech_pref in (rec.get("url") or ""):
                    return {"title": rec["title"], "url": rec["url"]}
    except Exception:
        pass

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
        is_ms = (
            "(m.s" in title or " m.s" in title or " m.s." in title or " ms" in title or
            "/ms" in url or "-ms/" in url or url.endswith("-ms/") or "-ms#" in url
        )
        is_undergrad = ("b.s" in title) or ("b.a" in title) or ("/bs" in url) or ("/ba" in url)
        return is_ms, is_phd, is_cert, is_undergrad

    def _degree_allowed(rec: Dict[str, str]) -> bool:
        is_ms, is_phd, is_cert, is_undergrad = _degree_flags(rec)
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

    # Respect blocklist unless explicitly mentioned
    allow_blocked = mentions_blocked_program(q_raw)
    def _not_blocked(rec: Dict[str, str]) -> bool:
        return (not is_blocked_program_url(rec.get("url", ""))) or allow_blocked

    filtered_pages = [rec for rec in _PROGRAM_PAGES if _not_blocked(rec)] if not (wants_ms or wants_phd or wants_cert or wants_undergrad) else None

    # Candidate pool respecting degree hints when present (blocklist-aware)
    if wants_ms or wants_phd or wants_cert or wants_undergrad:
        CANDS_ALL = [rec for rec in _PROGRAM_PAGES if _degree_allowed(rec)]
        if not mentions_blocked_program(q_raw):
            CANDS_ALL = [rec for rec in CANDS_ALL if not is_blocked_program_url(rec.get("url", ""))]
        if not CANDS_ALL:
            CANDS_ALL = [rec for rec in _PROGRAM_PAGES if mentions_blocked_program(q_raw) or not is_blocked_program_url(rec.get("url", ""))]
            if not CANDS_ALL:
                CANDS_ALL = list(_PROGRAM_PAGES)
    else:
        CANDS_ALL = filtered_pages if filtered_pages is not None else [rec for rec in _PROGRAM_PAGES if _not_blocked(rec)]

    # Matching paths
    for rec in CANDS_ALL:
        tnorm = rec.get("norm") or ""
        if not tnorm:
            continue
        if q_norm and (q_norm in tnorm or tnorm in q_norm):
            return {"title": rec["title"], "url": rec["url"]}

    def _slugify(s: str) -> str:
        s = re.sub(r"[^a-z0-9\s]", " ", (s or "").lower())
        s = re.sub(r"\s+", "-", s).strip("-")
        return s

    DROP = {"ms", "m.s", "m.s.", "program", "degree", "graduate", "master", "masters"}
    q_tokens = [tok for tok in q_norm.split() if tok and tok not in DROP]
    q_slug = _slugify(" ".join(q_tokens))
    if q_slug:
        for rec in CANDS_ALL:
            url = rec.get("url") or ""
            if q_slug in url:
                return {"title": rec["title"], "url": rec["url"]}

    q_tokens_set = set(q_norm.split())
    cands_overlap = [rec for rec in CANDS_ALL if (set((rec.get("norm") or "").split()) & q_tokens_set)]
    if not cands_overlap:
        cands_overlap = CANDS_ALL

    embed_model = get_embed_model()
    titles = [rec["title"] for rec in cands_overlap]
    vecs = embed_model.encode([q_raw] + titles, convert_to_numpy=True)
    qv, tv = vecs[0], vecs[1:]
    sims = (tv @ qv) / (np.linalg.norm(tv, axis=1) * np.linalg.norm(qv) + 1e-8)
    best_idx = int(np.argmax(sims))
    best = cands_overlap[best_idx]

    if wants_ms or wants_phd or wants_cert or wants_undergrad:
        is_ms, is_phd, is_cert, is_ug = _degree_flags(best)
        aligns = (
            (wants_ms and is_ms) or (wants_phd and is_phd) or (wants_cert and is_cert) or (wants_undergrad and is_ug)
        )
        if not aligns:
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

def update_section_stopwords(new_stopwords: List[str]) -> None:
    """Update the section stopwords used for filtering program pages."""
    global _SECTION_STOPWORDS
    if new_stopwords:
        _SECTION_STOPWORDS = tuple(new_stopwords)

def same_program_family(url1: str, url2: str) -> bool:
    def get_key(url: str) -> tuple:
        try:
            parts = [s for s in urlparse(url or "").path.split("/") if s]
            if "programs-study" not in parts:
                return ()
            
            idx = parts.index("programs-study")
            core = parts[idx : idx + 4]  # ['programs-study', school, program]
            
            if len(core) < 3:
                return ()
            
            return tuple(core[:3])
        except Exception:
            return ()
    
    k1 = get_key(url1)
    k2 = get_key(url2)
    
    return k1 != () and k1 == k2