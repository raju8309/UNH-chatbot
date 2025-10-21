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
                "norm": norm_title
            })
        except Exception:
            continue

    # Precompute embeddings for all program titles
    embed_model = get_embed_model()
    titles = [rec["title"] for rec in _PROGRAM_PAGES]
    _PROGRAM_EMBEDDINGS = embed_model.encode(titles, convert_to_numpy=True)
    print(f"Built program index with {len(_PROGRAM_PAGES)} programs and precomputed embeddings")

def match_program_alias(message: str) -> Optional[Dict[str, str]]:
    global _PROGRAM_EMBEDDINGS
    q_raw = (message or "").strip()
    # Filter candidates by degree intent
    degree_intent = _degree_intent(q_raw)
    if degree_intent["ms"] or degree_intent["phd"] or degree_intent["cert"]:
        candidates = [rec for rec in _PROGRAM_PAGES if _degree_allowed(degree_intent, rec)]
        result = _search_candidates(candidates, q_raw)
        if result:
            return result
    # Try full pool if filtered pool fails
    candidates = list(_PROGRAM_PAGES)
    result = _search_candidates(list(_PROGRAM_PAGES), q_raw)
    if result:
        return result
    return None

def _search_candidates(candidates: List[Dict[str, str]], q_raw: str) -> Optional[Dict[str, str]]:
    global _PROGRAM_EMBEDDINGS
    embed_model = get_embed_model()
    q_vec = embed_model.encode([q_raw], convert_to_numpy=True)[0]
    title_vecs = _PROGRAM_EMBEDDINGS[:len(candidates)]
    sims = (title_vecs @ q_vec) / (
        np.linalg.norm(title_vecs, axis=1) * np.linalg.norm(q_vec) + 1e-8
    )
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])
    if best_score >= 0.6:
        best = candidates[best_idx]
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