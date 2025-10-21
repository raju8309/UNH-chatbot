import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
import numpy as np
from config.settings import get_config, get_policy_terms
from models.ml_models import get_embed_model
from services.chunk_service import get_chunks_data, get_chunk_norms
from services.intent_service import is_admissions_url, is_degree_requirements_url, has_admissions_terms, has_policy_terms
from utils.course_utils import url_contains_course, title_starts_with_course, extract_title_leading_subject
from utils.program_utils import same_program_family

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

    # encode query
    q_vec = embed_model.encode([query], convert_to_numpy=True)[0]
    chunk_norms = get_chunk_norms()
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
    
    q_lower = (query or "").lower()
    allow_program = bool(alias_url)
    looks_policy = any(term in q_lower for term in policy_terms) or has_policy_terms(q_lower)
    looks_admissions = (intent_key == "admissions") or any(tok in q_lower for tok in [
        "admission", "admissions", "apply", "gre", "gmat", "test score", "test scores", "toefl", "ielts"
    ])

    # extract query terms
    query_terms = set(re.findall(r'\b\w+\b', q_lower))
    
    filtered: List[int] = []
    for i in cand_idxs:
        if i >= len(chunk_texts):
            continue

        chunk_text_lower = chunk_texts[i].lower()
        meta_i = chunk_meta[i] if i < len(chunk_meta) else {}
        tier = meta_i.get("tier", 2)

        # course queries → allow Tier 3 (preferred), skip only Tier 4 (program pages)
        if course_norm and tier == 4:
            continue

        # Strict same-subject filter for course-code queries
        # Keep only: (a) exact course search hits (Tier-2), (b) titles that start with the exact code,
        # or (c) titles whose leading subject (e.g., 'COMP') matches the asked subject.
        if course_norm and cfg.get("course_filters", {}).get("strict_subject_on_code", True):
            try:
                subj = course_norm.split()[0].upper()
                src_i = chunk_sources[i] if i < len(chunk_sources) else {}
                t_i = (src_i.get("title") or "")
                u_i = (src_i.get("url") or "")

                lead = extract_title_leading_subject(t_i) or ""
                url_has_exact = url_contains_course(u_i, course_norm)
                title_starts = title_starts_with_course(t_i, course_norm)

                if not (url_has_exact or title_starts or (lead == subj)):
                    continue
            except Exception:
                # Be conservative: if anything is odd, drop it
                continue

        if looks_policy:
            if tier == 3:
                if not has_policy_terms(chunk_text_lower):
                    continue
            if tier == 4:
                src_i = chunk_sources[i] if i < len(chunk_sources) else {}
                url_i = (src_i.get("url") or "")
                if alias_url:
                    if not same_program_family(url_i, alias_url):
                        continue
                    if not has_policy_terms(chunk_text_lower):
                        continue
                else:
                    continue
        if looks_admissions:
            if tier == 3 and not has_admissions_terms(chunk_text_lower):
                continue
        term_matches = len(query_terms.intersection(set(re.findall(r'\b\w+\b', chunk_text_lower))))
        if term_matches == 0 and sims[i] < 0.1:
            continue
        if course_norm and tier == 2:
            src_i = chunk_sources[i] if i < len(chunk_sources) else {}
            title_i = (src_i.get("title") or "")
            url_i = (src_i.get("url") or "")
            if not (url_contains_course(url_i, course_norm) or title_starts_with_course(title_i, course_norm)):
                continue
                
        # tier filtering
        if tier in (3, 4) and not allow_program:
            continue
        if tier == 4 and allow_program:
            src_i = chunk_sources[i] if i < len(chunk_sources) else {}
            if alias_url and same_program_family(src_i.get("url", ""), alias_url):
                pass
            else:
                if not _tier4_is_relevant_embed(query, i):
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
            same_prog_bonus = float(cfg.get("nudges", {}).get("same_program_bonus", 1.9))

        # course bonus → prefer Tier 3, allow Tier 2 fallback, penalize Tier 4
        course_bonus = 1.0
        if course_norm:
            url = (src_i.get("url") or "")
            title = (src_i.get("title") or "")
            nc = cfg.get("nudges", {}) or {}
            url_boost = float(nc.get("course_url_bonus", 1.9))
            title_boost = float(nc.get("course_title_bonus", 1.5))
            tier34_pen = float(nc.get("other_program_tier34_penalty", 0.25))
            if url_contains_course(url, course_norm):
                course_bonus = url_boost            # Tier 2 (search page) fallback
            elif title_starts_with_course(title, course_norm):
                course_bonus = title_boost          # Title matches course code
            elif tier == 3:
                course_bonus = 1.3                  # BOOST detailed course description
            elif tier == 4:
                course_bonus = tier34_pen           # penalize program pages
            else:
                course_bonus = 1.0

        admissions_bonus = 1.0
        if looks_admissions:
            url_i = (src_i.get("url") or "")
            txt_i = (chunk_texts[i] or "").lower()
            if is_admissions_url(url_i):
                admissions_bonus *= 1.6
            if has_admissions_terms(txt_i):
                admissions_bonus *= 1.25
            if is_degree_requirements_url(url_i):
                admissions_bonus *= 0.85

        credits_bonus = 1.0
        if intent_key == "degree_credits":
            txt_i = (chunk_texts[i] or "").lower()
            if re.search(r"\b\d{1,3}\b", txt_i) and ("credit" in txt_i):
                credits_bonus *= 1.4
                if re.search(r"\b(total|min(?:imum)?|required)\b", txt_i):
                    credits_bonus *= 1.2

        title_l = (src_i.get("title") or "").lower()
        section_bonus = 1.0
        if intent_key in ("degree_credits", "degree_requirements"):
            if alias_url and same_program_family(src_i.get("url", ""), alias_url):
                if ("degree requirements" in title_l) or re.search(r"\brequirements?\b", title_l):
                    section_bonus *= 1.6
            if any(s in title_l for s in ("career opportunities", "overview", "sample", "plan of study")):
                section_bonus *= 0.85

        rescored.append((i, base * nudge * same_prog_bonus * course_bonus * admissions_bonus * credits_bonus * section_bonus))

    rescored.sort(key=lambda x: x[1], reverse=True)
    ordered = [i for i, _ in rescored]

    if looks_policy:
        lead_t1 = None
        for i in ordered:
            if (chunk_meta[i] or {}).get("tier") == 1:
                lead_t1 = i
                break
        if lead_t1 is not None and ordered and ordered[0] != lead_t1:
            ordered.remove(lead_t1)
            ordered.insert(0, lead_t1)

    final: List[int] = []
    if looks_policy and bool(cfg.get("guarantees", {}).get("ensure_tier1_on_policy", True)):
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

    # Do not force Tier-4 program page for course queries
    want_program_page = bool(alias_url) and bool(cfg.get("guarantees", {}).get("ensure_tier4_on_program", True)) and (intent_key != "course_info")
    if want_program_page:
        def _is_t4(i: int) -> bool:
            return (chunk_meta[i] or {}).get("tier") == 4
        
        def _same_family_idx(i: int) -> bool:
            try:
                return same_program_family((chunk_sources[i] or {}).get("url", ""), alias_url or "")
            except Exception:
                return False

        has_tier4_same = any(_is_t4(i) and _same_family_idx(i) for i in final)

        if not has_tier4_same:
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

            if best_same[0] == -1:
                for j in range(len(chunk_meta)):
                    meta_j = chunk_meta[j] or {}
                    if meta_j.get("tier") != 4:
                        continue
                    if not _same_family_idx(j):
                        continue
                    sc = float(sims[j]) * _tier_boost(4)
                    if sc > best_same[1]:
                        best_same = (j, sc)

            inject = best_same[0] if best_same[0] != -1 else best_any[0]
            if inject != -1 and inject not in final:
                final.append(inject)
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

    # Ensure a course page appears by appending + prioritizing (Tier 2/3 only)
    if course_norm and not any(url_contains_course((chunk_sources[i] or {}).get("url", ""), course_norm) for i in final):
        best_course = (-1, -1.0)
        for i, score in rescored:
            if (chunk_meta[i] or {}).get("tier") == 4:
                continue
            if url_contains_course((chunk_sources[i] or {}).get("url", ""), course_norm) or \
               title_starts_with_course((chunk_sources[i] or {}).get("title", ""), course_norm):
                if score > best_course[1]:
                    best_course = (i, score)
        if best_course[0] != -1:
            final.append(best_course[0])
            seen_idx = set()
            dedup = []
            for ii in final:
                if ii not in seen_idx:
                    seen_idx.add(ii)
                    dedup.append(ii)
            if dedup and dedup[-1] != best_course[0]:
                try:
                    dedup.remove(best_course[0])
                    dedup.insert(0, best_course[0])
                except Exception:
                    pass
            final = dedup[:k]

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