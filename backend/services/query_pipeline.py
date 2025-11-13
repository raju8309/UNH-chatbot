"""
Simplified pipeline for preprocessing and retrieval logic.
Now supports direct answer return from gold set for high similarity matches.
"""
from services.qa_service import cached_answer_with_path
from services.query_transform_service import transform_query
from services.gold_set_service import get_gold_manager


def process_question_for_retrieval(incoming_message):
    """
    Main query processing pipeline with gold set direct answer support.
    
    Flow:
    1. Transform query if needed
    2. Check gold set for high similarity match
    3. If high similarity (>= threshold), return gold answer directly
    4. Otherwise, proceed with normal retrieval pipeline
    
    Args:
        incoming_message: The user's question (string or list of strings)
    
    Returns:
        dict with keys: answer, sources, retrieval_path, context, original_query, 
                       transformed_query, direct_gold_match (if applicable)
    """
    # handle list messages
    if isinstance(incoming_message, list):
        incoming_message = " ".join(incoming_message)
    
    original_query = incoming_message
    transformed_query = transform_query(original_query)
    if transformed_query != original_query:
        print(f"[QueryTransform] Original: {original_query} -> Transformed: {transformed_query}")

    # Check for direct gold answer match (high similarity)
    gold_manager = get_gold_manager()
    direct_match = gold_manager.get_direct_answer_with_similarity(transformed_query)
    
    if direct_match:
        answer, similarity, metadata = direct_match
        print(f"[GOLD SET] Using direct answer (similarity: {similarity:.3f})")
        
        # Format sources from gold metadata - make it look like normal sources
        sources = []
        if metadata.get('url'):
            # Extract a readable title from the gold_id or use a generic label
            gold_id = metadata.get('gold_id', 'unknown')
            url = metadata.get('url')
            
            # Create a clean title without "Gold Q&A:" prefix
            # Try to use category name in a friendly way
            category = metadata.get('category', '')
            if category:
                # Convert category like "academic-standards" to "Academic Standards"
                clean_category = category.replace('-', ' ').replace('_', ' ').title()
                title = f"{clean_category} Information"
            else:
                title = "Graduate Catalog"
            
            # Format like normal sources: "- Title (url)"
            sources.append(f"- {title} ({url})")
        
        # Create a minimal retrieval path showing this was a direct match
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
            answer=answer,
            sources=sources,
            retrieval_path=retrieval_path,
            context=f"Direct gold answer for: {metadata.get('gold_query')}",
            original_query=original_query,
            transformed_query=transformed_query,
            direct_gold_match=True,
            gold_metadata=metadata
        )
    
    # No high similarity match - proceed with normal retrieval
    answer, sources, retrieval_path, context = cached_answer_with_path(transformed_query)
    return dict(
        answer=answer,
        sources=sources,
        retrieval_path=retrieval_path,
        context=context,
        original_query=original_query,
        transformed_query=transformed_query,
        direct_gold_match=False
    )