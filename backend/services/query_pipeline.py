"""
Simplified pipeline for preprocessing and retrieval logic.
"""
from services.qa_service import cached_answer_with_path
from services.query_transform_service import transform_query
from config.settings import get_config

def process_question_for_retrieval(incoming_message):
    """
    NOTE: Intent detection, program level detection, and program alias matching 
    have been removed as they were broken and degrading retrieval quality.
    
    Args:
        incoming_message: The user's question (string or list of strings)
    
    Returns:
        dict with keys: answer, sources, retrieval_path, context, original_query, transformed_query
    """
    # handle list messages
    if isinstance(incoming_message, list):
        incoming_message = " ".join(incoming_message)
    
    original_query = incoming_message
    
    # Apply query transformation if enabled
    cfg = get_config()
    query_transform_config = cfg.get("query_transformation", {})
    transformed_query = original_query
    if query_transform_config.get("enabled", True):
        transformed_query = transform_query(original_query)
        if transformed_query != original_query:
            print(f"[QueryTransform] Original: {original_query} -> Transformed: {transformed_query}")

    # Simple retrieval without any scoping or intent-based modifications
    answer, sources, retrieval_path, context = cached_answer_with_path(transformed_query)
    return dict(
        answer=answer,
        sources=sources,
        retrieval_path=retrieval_path,
        context=context,
        original_query=original_query,
        transformed_query=transformed_query,
    )