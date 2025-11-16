from services.qa_service import cached_answer_with_path
from services.query_transform_service import transform_query
from services.gold_set_service import get_gold_manager
from config.settings import get_config

def process_question_for_retrieval(incoming_message):
    # handle list messages
    if isinstance(incoming_message, list):
        incoming_message = " ".join(incoming_message)
    
    original_query = incoming_message
    transformed_query = transform_query(original_query)
    if transformed_query != original_query:
        print(f"[QueryTransform] Original: {original_query} -> Transformed: {transformed_query}")

    # check configuration for dual answer mode
    cfg = get_config()
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
    
    # no high similarity match, proceed with normal retrieval
    answer, sources, retrieval_path, context = cached_answer_with_path(transformed_query)
    return dict(
        answer=answer,
        sources=sources,
        retrieval_path=retrieval_path,
        context=context,
        original_query=original_query,
        transformed_query=transformed_query,
        direct_gold_match=False,
        has_alternative=False,
        answer_mode="retrieval_only"
    )