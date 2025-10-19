import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
import numpy as np
from config.settings import get_config, get_policy_terms
from models.ml_models import get_embed_model
from services.chunk_service import get_chunks_data
from utils.course_utils import url_contains_course, title_starts_with_course
from utils.program_utils import same_program_family

def _program_intent(query: str) -> bool:
    cfg = get_config()
    q = (query or "").lower()   
    intent = cfg.get("intent", {})
    course_kw = intent.get("course_keywords", [])
    degree_kw = intent.get("degree_keywords", [])
    code_rx = intent.get("course_code_regex", r"\b[A-Z]{3,5}\s?\d{3}\b")   
    course_code = re.search(code_rx, query or "")
    return any(k in q for k in (course_kw + degree_kw)) or bool(course_code)

def _tier_boost(tier: int) -> float:
    cfg = get_config()
    return float(cfg.get("tier_boosts", {}).get(tier, 1.0))

def _is_acad_reg_url(url: str) -> bool:
    return isinstance(url, str) and "/graduate/academic-regulations-degree-requirements/" in url

def _title_for_sim(src: Dict[str, Any]) -> str:
    title = (src.get("title") or "").strip()
    url = src.get("url") or ""
    path = urlparse(url).path if url else ""
    segs = [s for s in path.split("/") if s]
    tail = " ".join(segs[-2:]) if segs else ""
    return (title + " " + tail).strip()

def _tier4_is_relevant_embed(query: str, idx: int, chunk_sources: List[Dict]) -> bool:
    cfg = get_config()
    gate = cfg.get("tier4_gate", {})
    
    if not gate.get("use_embedding", True):
        return True
    
    if idx >= len(chunk_sources):
        return False
    
    cand = _title_for_sim(chunk_sources[idx])
    if not cand:
        return False
    
    embed_model = get_embed_model()
    q_vec, c_vec = embed_model.encode([query, cand], convert_to_numpy=True)
    
    sim = float(
        np.dot(q_vec, c_vec) /
        (np.linalg.norm(q_vec) * np.linalg.norm(c_vec) + 1e-8)
    )
    
    thresh = float(gate.get("min_title_sim", 0.42))
    return sim >= thresh

def _apply_guarantees(
    ordered: List[int],
    k: int,
    looks_policy: bool,
    alias_url: Optional[str],
    course_norm: Optional[str],
    chunk_meta: List[Dict],
    chunk_sources: List[Dict],
    rescored: List[tuple],
    sims,
) -> List[int]:
    cfg = get_config()
    final: List[int] = []
    
    # guarantee 1: Ensure tier-1 on policy questions
    if looks_policy and cfg.get("guarantees", {}).get("ensure_tier1_on_policy", True):
        has_tier1 = any((chunk_meta[i] or {}).get("tier") == 1 for i in ordered[:k])
        
        if not has_tier1:
            best_t1_idx = -1
            best_t1_score = -1.0
            
            for i in range(len(chunk_meta)):
                meta_i = chunk_meta[i] or {}
                if meta_i.get("tier") == 1:
                    sc = float(sims[i]) * 1.35 * 1.15  # tier boost * policy nudge
                    if sc > best_t1_score:
                        best_t1_score = sc
                        best_t1_idx = i
            
            if best_t1_idx != -1:
                final.append(best_t1_idx)
    
    # add remaining from ordered list
    for i in ordered:
        if len(final) >= k:
            break
        if i not in final:
            final.append(i)
    
    final = final[:k]
    
    # guarantee 2: Ensure tier-4 same-program when alias is set
    want_program_page = (
        bool(alias_url) and
        cfg.get("guarantees", {}).get("ensure_tier4_on_program", True)
    )
    
    if want_program_page:
        def _is_t4(i: int) -> bool:
            return (chunk_meta[i] or {}).get("tier") == 4
        
        def _same_family_idx(i: int) -> bool:
            try:
                return same_program_family(
                    (chunk_sources[i] or {}).get("url", ""),
                    alias_url or ""
                )
            except Exception:
                return False
        
        has_tier4_same = any(_is_t4(i) and _same_family_idx(i) for i in final)
        
        if not has_tier4_same:
            # find best same-family tier-4
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
            
            # widen search to all chunks if needed
            if best_same[0] == -1:
                for j in range(len(chunk_meta)):
                    meta_j = chunk_meta[j] or {}
                    if meta_j.get("tier") != 4:
                        continue
                    if not _same_family_idx(j):
                        continue
                    sc = float(sims[j]) * 1.0  # tier 4 boost
                    if sc > best_same[1]:
                        best_same = (j, sc)
            
            inject = best_same[0] if best_same[0] != -1 else best_any[0]
            
            if inject != -1 and inject not in final:
                final.append(inject)
                
                # deduplicate and trim
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
    
    # guarantee 3: Ensure course page when asking about a course
    if course_norm and not any(
        url_contains_course((chunk_sources[i] or {}).get("url", ""), course_norm)
        for i in final
    ):
        best_course = (-1, -1.0)
        for i, score in rescored:
            if url_contains_course((chunk_sources[i] or {}).get("url", ""), course_norm):
                if score > best_course[1]:
                    best_course = (i, score)
        
        if best_course[0] != -1:
            final[-1] = best_course[0]
    
    return final

def search_chunks(
    query: str,
    topn: int = 40,
    k: int = 5,
    alias_url: Optional[str] = None,
    intent_key: Optional[str] = None,
    course_norm: Optional[str] = None,
) -> Tuple[List[int], List[Dict[str, Any]]]:
    cfg = get_config()
    policy_terms = get_policy_terms()
    embed_model = get_embed_model()
    
    chunks_embeddings, chunk_texts, chunk_sources, chunk_meta = get_chunks_data()
    
    if chunks_embeddings is None or not chunk_texts:
        return [], []
    
    # encode query
    q_vec = embed_model.encode([query], convert_to_numpy=True)[0]
    
    # compute cosine similarities
    chunk_norms = np.linalg.norm(chunks_embeddings, axis=1)
    query_norm = np.linalg.norm(q_vec)
    
    valid_chunks = chunk_norms > 1e-8
    sims = np.zeros(len(chunks_embeddings))
    
    if query_norm > 1e-8:
        sims[valid_chunks] = (
            (chunks_embeddings[valid_chunks] @ q_vec) /
            (chunk_norms[valid_chunks] * query_norm)
        )
    
    # get top candidates
    cand_idxs = np.argsort(-sims)[:topn * 2].tolist()
    
    # enhanced filtering
    q_lower = (query or "").lower()
    allow_program = _program_intent(query) or bool(alias_url)
    looks_policy = any(term in q_lower for term in policy_terms)
    
    # extract query terms
    query_terms = set(re.findall(r'\b\w+\b', q_lower))
    
    filtered: List[int] = []
    for i in cand_idxs:
        if i >= len(chunk_texts):
            continue
        
        chunk_text_lower = chunk_texts[i].lower()
        meta_i = chunk_meta[i] if i < len(chunk_meta) else {}
        tier = meta_i.get("tier", 2)
        
        # relevance check
        term_matches = len(
            query_terms.intersection(set(re.findall(r'\b\w+\b', chunk_text_lower)))
        )
        if term_matches == 0 and sims[i] < 0.1:
            continue
        
        # tier filtering
        if tier in (3, 4) and not allow_program:
            continue
        
        if tier == 4 and allow_program:
            src_i = chunk_sources[i] if i < len(chunk_sources) else {}
            if alias_url and same_program_family(src_i.get("url", ""), alias_url):
                pass  # Keep it
            else:
                if not _tier4_is_relevant_embed(query, i, chunk_sources):
                    continue
        
        filtered.append(i)
    
    # fallback if no results
    if not filtered:
        allowed_tiers = {1, 2} if not allow_program else {1, 2, 3, 4}
        filtered = [
            i for i in range(len(chunk_meta))
            if ((chunk_meta[i] or {}).get("tier") in allowed_tiers)
        ] or list(range(len(chunk_meta)))
    
    # rescore with bonuses
    policy_nudge = float(cfg.get("nudges", {}).get("policy_acadreg_url", 1.15))
    
    rescored = []
    for i in filtered:
        meta_i = chunk_meta[i] if i < len(chunk_meta) else {}
        src_i = chunk_sources[i] if i < len(chunk_sources) else {}
        tier = meta_i.get("tier", 2)
        base = float(sims[i]) * _tier_boost(tier)
        
        # policy nudge
        nudge = policy_nudge if looks_policy and _is_acad_reg_url(src_i.get("url", "")) else 1.0
        
        # same program bonus
        same_prog_bonus = 1.0
        if alias_url and same_program_family(src_i.get("url", ""), alias_url):
            same_prog_bonus = 1.9
        
        # course bonus
        course_bonus = 1.0
        if course_norm:
            url = src_i.get("url") or ""
            title = src_i.get("title") or ""
            if url_contains_course(url, course_norm):
                course_bonus = 1.8
            elif title_starts_with_course(title, course_norm):
                course_bonus = 1.4
            elif tier in (3, 4):
                course_bonus = 0.9
            elif tier == 1 and looks_policy:
                course_bonus = 1.0
        
        rescored.append((i, base * nudge * same_prog_bonus * course_bonus))
    
    rescored.sort(key=lambda x: x[1], reverse=True)
    ordered = [i for i, _ in rescored]
    
    # build preliminary top-k with guarantees
    final = _apply_guarantees(
        ordered, k, looks_policy, alias_url, course_norm,
        chunk_meta, chunk_sources, rescored, sims
    )
    
    # build retrieval path
    retrieval_path = []
    for rank, i in enumerate(final, start=1):
        src = chunk_sources[i] if i < len(chunk_sources) else {}
        meta = chunk_meta[i] if i < len(chunk_meta) else {}
        retrieval_path.append({
            "rank": rank,
            "idx": i,
            "score": round(float(sims[i]), 6),
            "title": src.get("title"),
            "url": src.get("url"),
            "tier": meta.get("tier"),
            "tier_name": meta.get("tier_name"),
            "text": chunk_texts[i] if i < len(chunk_texts) else ""
        })
    
    return final, retrieval_path