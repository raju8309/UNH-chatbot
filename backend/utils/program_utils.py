import re
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
import numpy as np
from models.ml_models import get_embed_model
from config.settings import get_config

# in-memory index of program pages and their embeddings
_PROGRAM_PAGES: List[Dict[str, str]] = []
_PROGRAM_EMBEDDINGS = None

# section stopwords to filter out
_SECTION_STOPWORDS = (
    "upon completion", "program learning outcomes", "admission requirements",
    "requirements", "application requirements", "core courses", "electives",
    "overview", "policies", "sample", "plan of study"
)

# tokens that are too generic to be useful for alias overlap
_GENERIC_TOKENS = {
    "program", "programs", "degree", "degrees", "graduate", "graduation",
    "studies", "study", "school", "college", "university", "science", "sciences"
}
# tokens that change the meaning of a program (apply penalties if absent in query)
_SPECIAL_TERMS_STRONG = {"management"}
_SPECIAL_TERMS = {"policy", "risk", "engineering", "biomedical", "biotechnology", "computational", "applied"}

def normalize_text(text: Optional[str]) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Helper: tokenize for match
def _tokenize_for_match(s: str) -> List[str]:
    s = normalize_text(s)
    # keep meaningful program tokens; drop tiny words
    toks = [t for t in s.split() if len(t) >= 3]
    return toks

def _url_tokens(url: str) -> List[str]:
    try:
        path = urlparse(url or "").path.lower()
        parts = [p for p in re.split(r"[/\-\._]", path) if p]
        # ignore common catalog scaffolding terms
        stop = {"graduate", "programs", "study", "catalog", "unh", "edu"}
        return [p for p in parts if p and p not in stop and len(p) >= 3]
    except Exception:
        return []

def looks_like_program_url(url: str) -> bool:
    return (
        "/graduate/programs-study/" in url
        and "/search/?" not in url
        and "/academic-regulations-degree-requirements/" not in url
    )

def build_program_index(chunk_sources: List[Dict], chunk_meta: List[Dict]) -> None:
    global _PROGRAM_EMBEDDINGS
    _PROGRAM_PAGES.clear()
    _PROGRAM_EMBEDDINGS = None

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

            _PROGRAM_PAGES.append({
                "title": title,
                "url": url,
                "norm": norm_title,
                "vec": None,
                "tokens": list(set(_tokenize_for_match(title) + _url_tokens(url))),
            })
        except Exception:
            continue

    # Precompute embeddings and attach to each record
    embed_model = get_embed_model()
    titles = [rec["title"] for rec in _PROGRAM_PAGES]
    vecs = embed_model.encode(titles, convert_to_numpy=True)
    for rec, v in zip(_PROGRAM_PAGES, vecs):
        rec["vec"] = v.astype(np.float32)

    print(f"Built program index with {len(_PROGRAM_PAGES)} programs and precomputed embeddings")

def match_program_alias(message: str) -> Optional[Dict[str, str]]:
    q_raw = (message or "").strip()
    cfg = get_config() or {}
    # Filter candidates by degree intent
    degree_intent = _degree_intent(q_raw)
    if degree_intent["ms"] or degree_intent["phd"] or degree_intent["cert"]:
        candidates = [rec for rec in _PROGRAM_PAGES if _degree_allowed(degree_intent, rec)]
        result = _search_candidates(candidates, q_raw, degree_intent)
        if result:
            return result
    # Try full pool if filtered pool fails
    return _search_candidates(list(_PROGRAM_PAGES), q_raw, degree_intent)

def _search_candidates(candidates: List[Dict[str, str]], q_raw: str, degree_intent: Optional[Dict[str, bool]] = None) -> Optional[Dict[str, str]]:
    if not candidates:
        return None
    cfg = get_config() or {}
    strict_alias = bool(cfg.get("STRICT_PROGRAM_ALIAS", True))
    min_cos = float(cfg.get("ALIAS_MIN_COS", 0.68))
    min_cos_olap = float(cfg.get("ALIAS_MIN_COS_WITH_OVERLAP", 0.62))
    min_olap = int(cfg.get("ALIAS_MIN_OVERLAP", 1))
    # encode query
    embed_model = get_embed_model()
    q_vec = embed_model.encode([q_raw], convert_to_numpy=True)[0].astype(np.float32)
    q_norm = np.linalg.norm(q_vec) + 1e-8
    q_toks_all = set(_tokenize_for_match(q_raw))
    # Build candidate matrix
    cand_vecs = [c.get("vec") for c in candidates if c.get("vec") is not None]
    if not cand_vecs:
        return None
    title_vecs = np.vstack(cand_vecs)
    norms = np.linalg.norm(title_vecs, axis=1) + 1e-8
    cos = (title_vecs @ q_vec) / (norms * q_norm)
    # Token overlap (specific only; ignore generic tokens)
    cand_tok_sets = [set(c.get("tokens", [])) for c in candidates]
    overlaps_specific = np.array([
        len((q_toks_all - _GENERIC_TOKENS).intersection(tokset - _GENERIC_TOKENS))
        for tokset in cand_tok_sets
    ], dtype=np.float32)
    # Gentle penalties if candidate has a specific modifier not present in query
    penalties = []
    for tokset in cand_tok_sets:
        p = 0.0
        # strong penalty for 'management' when user didn't say it
        if ("management" in tokset) and ("management" not in q_toks_all):
            p -= 0.10
        # other special terms
        for t in (_SPECIAL_TERMS & tokset):
            if t not in q_toks_all:
                p -= 0.05
        penalties.append(p)
    penalties = np.array(penalties, dtype=np.float32)

    # Query-specific pairwise adjustments for common confusions (gentle nudges)
    ql = q_raw.lower()
    pairwise_adj = []
    for rec, tokset in zip(candidates, cand_tok_sets):
        adj = 0.0
        title_l = (rec.get("title") or "").lower()
        # IT vs IT-Management: if user says "information technology" but not "management",
        # downweight candidates that contain "management"
        if ("information technology" in ql) and ("management" not in ql) and ("management" in tokset):
            adj -= 0.08
        # Biotech vs MCBS: if user says "biotechnology", nudge away titles that look like MCBS without "biotechnology"
        if ("biotechnology" in ql) and ("biotechnology" not in tokset) and (("molecular" in tokset) or ("cellular" in tokset)):
            adj -= 0.05
        pairwise_adj.append(adj)
    pairwise_adj = np.array(pairwise_adj, dtype=np.float32)

    # Combined score (cosine + small nudge from overlap + penalties + pairwise_adj)
    combined = 0.90 * cos + 0.10 * (np.minimum(overlaps_specific, 3) / 3.0) + penalties + pairwise_adj
    best_idx = int(np.argmax(combined))
    best_cos = float(cos[best_idx])
    best_overlap = int(overlaps_specific[best_idx])
    best = candidates[best_idx]
    # Acceptance rules
    if strict_alias:
        di = degree_intent or {"ms": False, "phd": False, "cert": False}
        degree_cue = di["ms"] or di["phd"] or di["cert"]
        # only accept if: strong cosine OR decent cosine with enough specific overlap OR explicit degree cue
        if (best_cos >= min_cos) or (best_cos >= min_cos_olap and best_overlap >= min_olap) or degree_cue:
            return {"title": best["title"], "url": best["url"]}
        return None
    # non-strict (legacy) behavior
    if best_cos >= 0.68 or (best_cos >= 0.62 and best_overlap >= 1):
        return {"title": best["title"], "url": best["url"]}
    return None

def _degree_flags(rec: Dict[str, str]) -> Tuple[bool, bool, bool]:
    title = (rec.get("title") or "").lower()
    url = (rec.get("url") or "").lower()
    is_cert = "certificate" in title or "certificate" in url
    is_phd = "phd" in title or "ph.d" in title or "/phd" in url
    is_ms = "m.s" in title or " ms" in title or "/ms" in url or "-ms/" in url
    return is_ms, is_phd, is_cert

def _degree_allowed(intent: Dict[str, bool], rec: Dict[str, str]) -> bool:
    is_ms, is_phd, is_cert = _degree_flags(rec)
    if intent["ms"] and not is_ms:
        return False
    if intent["phd"] and not is_phd:
        return False
    if intent["cert"] and not is_cert:
        return False
    return True

def _degree_intent(message: str) -> Dict[str, bool]:
    t = f" {message.lower()} "
    wants_ms = bool(re.search(r"\bms\b|\bm\.s\.?\b|master'?s", t))
    wants_phd = bool(re.search(r"\bphd\b|ph\.d\.?\b|doctoral|doctorate", t))
    wants_cert = "certificate" in t
    return {"ms": wants_ms, "phd": wants_phd, "cert": wants_cert}

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