import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
import numpy as np
from config.settings import get_config, get_policy_terms
from models.ml_models import get_embed_model
from services.chunk_service import get_chunks_data, get_chunk_norms
from services.intent_service import (
    is_admissions_url,
    is_degree_requirements_url,
    has_admissions_terms,
    has_policy_terms
)
from utils.course_utils import (
    url_contains_course,
    title_starts_with_course,
    extract_title_leading_subject
)
from utils.program_utils import same_program_family
from services.gold_set_service import get_gold_manager

def _tier_boost(tier: int) -> float:
    cfg = get_config()
    
    # If gold set is disabled, treat Tier 0 chunks as Tier 2 (general info)
    if tier == 0:
        gold_enabled = cfg.get("gold_set", {}).get("enabled", True)
        if not gold_enabled:
            tier = 2  # Demote gold chunks to regular tier
        else:
            return float(cfg.get("gold_set", {}).get("tier_boost", 3.0))
    
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

def _tier4_is_relevant_embed(query: str, idx: int) -> bool:
    cfg = get_config()
    gate = cfg.get("tier4_gate", {})
    if not gate.get("use_embedding", True):
        return True
    _, _, chunk_sources, _ = get_chunks_data()
    if idx >= len(chunk_sources):
        return False
    cand = _title_for_sim(chunk_sources[idx])
    if not cand:
        return False
    embed_model = get_embed_model()
    q_vec, c_vec = embed_model.encode([query, cand], convert_to_numpy=True)
    sim = float(np.dot(q_vec, c_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(c_vec) + 1e-8))
    thresh = float(gate.get("min_title_sim", 0.42))
    return sim >= thresh

def _apply_gold_boost(
    rescored: List[Tuple[int, float]],
    query: str,
    chunk_texts: List[str],
    chunk_sources: List[Dict],
    chunk_meta: List[Dict]
) -> List[Tuple[int, float]]:
    # Check if gold set is enabled
    cfg = get_config()
    gold_enabled = cfg.get("gold_set", {}).get("enabled", True)
    
    if not gold_enabled:
        return rescored  # Skip all gold boosting if disabled
    
    try:
        gold_manager = get_gold_manager()
    except:
        return rescored  # Gold manager not available
    
    try:
        gold_match = gold_manager.find_matching_gold_entry(query, threshold=0.85)
    except:
        gold_match = None

    boosted_rescored = []
    for idx, score in rescored:
        if idx >= len(chunk_meta):
            boosted_rescored.append((idx, score))
            continue
        
        meta = chunk_meta[idx]
        text = chunk_texts[idx] if idx < len(chunk_texts) else ""
        
        try:
            gold_boost = gold_manager.get_gold_boost_for_chunk(text, meta, query)
        except:
            gold_boost = 1.0

        if gold_match and gold_boost > 1.0:
            gold_id = meta.get('gold_id', '')
            match_id = gold_match.get('id', '')
            if gold_id == match_id:
                gold_boost *= 2.0
                print(f"Exact gold match found: {gold_id}")

        boosted_rescored.append((idx, score * gold_boost))
    
    return boosted_rescored

def search_chunks(
    query: str,
    topn: int = 40,
    k: int = 5,
    alias_url: Optional[str] = None,
    intent_key: Optional[str] = None,
    course_norm: Optional[str] = None
) -> Tuple[List[int], List[Dict[str, Any]]]:

    cfg = get_config()
    policy_terms = get_policy_terms()
    embed_model = get_embed_model()
    chunks_embeddings, chunk_texts, chunk_sources, chunk_meta = get_chunks_data()

    if chunks_embeddings is None or not chunk_texts:
        return [], []

    # Check if gold set is enabled
    gold_enabled = cfg.get("gold_set", {}).get("enabled", True)
    
    # Only attempt direct gold matching if gold set is enabled
    if gold_enabled:
        gold_manager = get_gold_manager()
        direct_match = gold_manager.find_matching_gold_entry(query, threshold=0.95)
        if direct_match:
            gold_id = direct_match.get('id')
            for idx, meta in enumerate(chunk_meta):
                if meta.get('gold_id') == gold_id:
                    retrieval_path = [{
                        "rank": 1,
                        "idx": idx,
                        "score": direct_match.get('match_score'),
                        "title": f"Gold Q&A: {gold_id}",
                        "url": direct_match.get('url', ''),
                        "tier": 0,
                        "tier_name": "gold_set",
                        "text": chunk_texts[idx] if idx < len(chunk_texts) else "",
                        "gold_match": True,
                        "match_score": direct_match.get('match_score')
                    }]
                    return [idx], retrieval_path

    q_vec = embed_model.encode([query], convert_to_numpy=True)[0]
    chunk_norms = get_chunk_norms()
    query_norm = np.linalg.norm(q_vec)
    valid_chunks = chunk_norms > 1e-8
    sims = np.zeros(len(chunks_embeddings))

    if query_norm > 1e-8:
        sims[valid_chunks] = (chunks_embeddings[valid_chunks] @ q_vec) / (
            chunk_norms[valid_chunks] * query_norm
        )

    cand_idxs = np.argsort(-sims)[:topn * 2].tolist()

    filtered = cand_idxs  # always use top candidates
    rescored = []
    for i in filtered:
        meta_i = chunk_meta[i] if i < len(chunk_meta) else {}
        base = float(sims[i]) * _tier_boost(meta_i.get("tier", 2))
        rescored.append((i, base))

    # Apply gold boosting only if enabled
    rescored = _apply_gold_boost(rescored, query, chunk_texts, chunk_sources, chunk_meta)
    rescored.sort(key=lambda x: x[1], reverse=True)
    ordered = [i for i, _ in rescored]

    # Ensure gold in results only if gold set is enabled
    gold_cfg = cfg.get("gold_set", {})
    if gold_enabled and gold_cfg.get("ensure_gold_in_results", True):
        top_k = ordered[:k]
        has_gold = any((chunk_meta[i] or {}).get("tier") == 0 for i in top_k if i < len(chunk_meta))
        if not has_gold:
            best_gold_idx = None
            best_gold_score = -1.0
            for idx, score in rescored:
                if idx < len(chunk_meta) and (chunk_meta[idx] or {}).get("tier") == 0:
                    if score > best_gold_score:
                        best_gold_score = score
                        best_gold_idx = idx
            if best_gold_idx is not None and best_gold_score > 0.3:
                ordered = [i for i in ordered if i != best_gold_idx]
                ordered.insert(1, best_gold_idx)
                print("Ensured gold chunk in results at position 2")

    final = ordered[:k]

    retrieval_path = []
    for rank, i in enumerate(final, start=1):
        src = chunk_sources[i] if i < len(chunk_sources) else {}
        meta = chunk_meta[i] if i < len(chunk_meta) else {}
        entry = {
            "rank": rank,
            "idx": i,
            "score": round(float(sims[i]), 6),
            "title": src.get("title"),
            "url": src.get("url"),
            "tier": meta.get("tier"),
            "tier_name": meta.get("tier_name"),
            "text": chunk_texts[i] if i < len(chunk_texts) else ""
        }
        if meta.get("is_gold", False):
            entry["is_gold"] = True
            entry["gold_id"] = meta.get("gold_id")
        retrieval_path.append(entry)

    return final, retrieval_path