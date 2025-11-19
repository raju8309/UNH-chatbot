from services.qa_service import cached_answer_with_path
from services.query_transform_service import transform_query
from services.gold_set_service import get_gold_manager
from config.settings import get_config
from config import settings

# --- scoring helper for retrieval fusion ---
def _score_retrieval(result_dict, query_text: str) -> float:
    """
    Assign a simple quality score to a cached_answer_with_path() result.
    Heuristics:
      +1 if there are any sources
      +len(sources)*0.1
      +len(retrieval_path)*0.05
      +0.5 if every content token from query appears at least once in answer (loose check)
    """
    import re as _re
    answer = result_dict.get("answer")
    sources = result_dict.get("sources") or []
    retrieval_path = result_dict.get("retrieval_path") or []
    score = 0.0
    if sources:
        score += 1.0 + 0.1 * len(sources)
    if retrieval_path:
        score += 0.05 * len(retrieval_path)
    toks = [t for t in _re.findall(r"[a-zA-Z]+", (query_text or "").lower())
            if t not in {"the","a","an","in","for","to","of","and","or","at","on","vs","is","are","what","how","can","do","does"}]
    if toks:
        ans_low = (answer or "").lower()
        hits = sum(1 for t in set(toks) if t in ans_low)
        if hits == len(set(toks)):
            score += 0.5
    return score

# --- clarifier backoff rules (retrieval-only enrichments) ---
_CLARIFIER_RULES = {
    "failing": "failing grade",
    "probation": "academic probation status",
    "dismissal": "academic dismissal policy",
    "add/drop": "add/drop deadline",
    "add drop": "add/drop deadline",
    "overload": "overload permission",
    "leave of absence": "leave of absence policy",
    "withdrawal": "withdrawal policy",
    "registration hold": "registration hold reasons",
    "assistantship": "graduate assistantship eligibility",
    "ra ta": "RA/TA tuition waiver",
    "visa": "visa full-time enrollment requirement",
}

def _clarify_for_retrieval(original: str) -> str | None:
    """Build a slightly enriched query for retrieval only, without changing user-visible text."""
    if not original:
        return None
    text = original
    low = original.lower()
    changed = False

    for key, expansion in _CLARIFIER_RULES.items():
        if key in low and expansion.lower() not in low:
            # simple strategy: append expansion phrase for retrieval
            text = text + " " + expansion
            changed = True

    if not changed or text.strip() == original.strip():
        return None
    return text

def process_question_for_retrieval(incoming_message):
    # handle list messages
    if isinstance(incoming_message, list):
        incoming_message = " ".join(incoming_message)
    
    # Apply query transformation before intent detection and retrieval
    cfg = get_config()
    query_transform_config = cfg.get("query_transformation", {})
    original_query = incoming_message
    user_query = incoming_message
    transformed_query = user_query
    if query_transform_config.get("enabled", True):
        transformed_query = transform_query(user_query)
        if transformed_query != user_query:
            print(f"[QueryTransform] Original: {user_query} -> Transformed: {transformed_query}")
        user_query = transformed_query

    # clarifier backoff: if no effective rewrite happened, build a retrieval-only clarified query
    clarified_query = None
    if transformed_query == original_query:
        clarified_query = _clarify_for_retrieval(original_query)

    # check configuration for dual answer mode
    gold_cfg = cfg.get("gold_set", {})
    dual_mode = gold_cfg.get("enable_dual_answers", True)
    
    # check for direct gold answer match
    gold_manager = get_gold_manager()
    direct_match = gold_manager.get_direct_answer_with_similarity(transformed_query)
    
    if direct_match:
        answer_gold, similarity, metadata = direct_match
        print(f"[GOLD SET] Match found (similarity: {similarity:.3f})")
        
        if dual_mode:
            print(f"[DUAL MODE] Generating both gold and retrieval answers...")
            
            # generate retrieval based answer as alternative
            answer_retrieval, sources_retrieval, retrieval_path, context = cached_answer_with_path(transformed_query)
            
            # format gold sources
            sources_gold = []
            if metadata.get('url'):
                category = metadata.get('category', '')
                if category:
                    clean_category = category.replace('-', ' ').replace('_', ' ').title()
                    title = f"{clean_category} Information"
                else:
                    title = "Graduate Catalog"
                sources_gold.append(f"- {title} ({metadata.get('url')})")
            
            # create retrieval path for gold
            gold_retrieval_path = [{
                "rank": 1,
                "gold_id": metadata.get('gold_id'),
                "gold_query": metadata.get('gold_query'),
                "similarity_score": similarity,
                "match_type": "direct_gold_match",
                "url": metadata.get('url'),
                "tier": 0,
                "tier_name": "gold_set_direct"
            }]
            
            return dict(
                # primary answer (gold)
                answer=answer_gold,
                sources=sources_gold,
                retrieval_path=gold_retrieval_path,
                context=f"Direct gold answer for: {metadata.get('gold_query')}",
                original_query=original_query,
                transformed_query=transformed_query,
                direct_gold_match=True,
                gold_metadata=metadata,
                
                # Alternative answer (retrieval)
                has_alternative=True,
                alternative_answer=answer_retrieval,
                alternative_sources=sources_retrieval,
                alternative_retrieval_path=retrieval_path,
                alternative_context=context,
                alternative_type="retrieval",
                
                # Metadata for frontend
                answer_mode="dual",
                gold_similarity=similarity
            )
        else:
            # dual mode disabled return only gold answer
            print(f"[GOLD SET] Using direct answer only (dual mode disabled)")
            
            sources = []
            if metadata.get('url'):
                category = metadata.get('category', '')
                if category:
                    clean_category = category.replace('-', ' ').replace('_', ' ').title()
                    title = f"{clean_category} Information"
                else:
                    title = "Graduate Catalog"
                sources.append(f"- {title} ({metadata.get('url')})")
            
            retrieval_path = [{
                "rank": 1,
                "gold_id": metadata.get('gold_id'),
                "gold_query": metadata.get('gold_query'),
                "similarity_score": similarity,
                "match_type": "direct_gold_match",
                "url": metadata.get('url'),
                "tier": 0,
                "tier_name": "gold_set_direct"
            }]
            
            return dict(
                answer=answer_gold,
                sources=sources,
                retrieval_path=retrieval_path,
                context=f"Direct gold answer for: {metadata.get('gold_query')}",
                original_query=original_query,
                transformed_query=transformed_query,
                direct_gold_match=True,
                gold_metadata=metadata,
                has_alternative=False,
                answer_mode="gold_only"
            )
    
    # --- retrieval (single or dual) ---
    if getattr(settings, "RETRIEVAL_USE_DUAL_QUERY", False):
        # Run original (pre-transform) and rewritten (post-transform)
        ans_a, src_a, path_a, ctx_a = cached_answer_with_path(original_query)
        ans_b, src_b, path_b, ctx_b = cached_answer_with_path(user_query)

        candidates = []
        res_a = dict(answer=ans_a, sources=src_a, retrieval_path=path_a, context=ctx_a)
        res_b = dict(answer=ans_b, sources=src_b, retrieval_path=path_b, context=ctx_b)
        candidates.append((res_a, original_query))
        candidates.append((res_b, user_query))

        # Optional third candidate: clarified original query (retrieval-only enrichment)
        if clarified_query and clarified_query not in {original_query, user_query}:
            ans_c, src_c, path_c, ctx_c = cached_answer_with_path(clarified_query)
            res_c = dict(answer=ans_c, sources=src_c, retrieval_path=path_c, context=ctx_c)
            candidates.append((res_c, clarified_query))

        best_score = float("-inf")
        chosen = None
        for res, q in candidates:
            score = _score_retrieval(res, q)
            if score > best_score:
                best_score = score
                chosen = res
    else:
        # Single-query mode: prefer a clarified query if we have one, otherwise use transformed query
        effective_query = clarified_query or user_query
        ans_b, src_b, path_b, ctx_b = cached_answer_with_path(effective_query)
        chosen = dict(answer=ans_b, sources=src_b, retrieval_path=path_b, context=ctx_b)

    return dict(
        answer=chosen["answer"],
        sources=chosen["sources"],
        retrieval_path=chosen["retrieval_path"],
        context=chosen["context"],
        original_query=original_query,
        transformed_query=transformed_query,
        direct_gold_match=False,
        has_alternative=False,
        answer_mode="retrieval_only"
    )