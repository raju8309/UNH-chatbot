import re
from typing import Dict, List, Optional
from urllib.parse import urlparse
import numpy as np
from models.ml_models import get_embed_model

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

def match_program_alias(message: str) -> Optional[Dict[str, str]]:
    if not _PROGRAM_PAGES:
        return None
    
    q_raw = (message or "").strip()
    q_norm = normalize_text(q_raw)
    
    # fast path: exact title containment
    for rec in _PROGRAM_PAGES:
        tnorm = rec["norm"]
        if not tnorm:
            continue
        if q_norm and (q_norm in tnorm or tnorm in q_norm):
            return {"title": rec["title"], "url": rec["url"]}
    
    # try URL slug matching
    drop_words = {"ms", "m.s", "m.s.", "program", "degree", "graduate", "master", "masters"}
    q_tokens = [t for t in q_norm.split() if t and t not in drop_words]
    q_slug = "-".join(q_tokens)
    
    if q_slug:
        for rec in _PROGRAM_PAGES:
            url = rec.get("url") or ""
            if q_slug in url:
                return {"title": rec["title"], "url": rec["url"]}
    
    # fallback: embedding-based similarity
    q_tokens_set = set(q_norm.split())
    candidates = [
        rec for rec in _PROGRAM_PAGES
        if (set(rec["norm"].split()) & q_tokens_set)
    ]
    
    if not candidates:
        candidates = _PROGRAM_PAGES
    
    embed_model = get_embed_model()
    titles = [rec["title"] for rec in candidates]
    vecs = embed_model.encode([q_raw] + titles, convert_to_numpy=True)
    
    q_vec = vecs[0]
    title_vecs = vecs[1:]
    
    sims = (title_vecs @ q_vec) / (
        np.linalg.norm(title_vecs, axis=1) * np.linalg.norm(q_vec) + 1e-8
    )
    
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])
    
    if best_sim < 0.45:
        return None
    
    best = candidates[best_idx]
    return {"title": best["title"], "url": best["url"]}


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