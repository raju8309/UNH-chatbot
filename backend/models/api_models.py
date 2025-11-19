from typing import Any, Dict, List, Optional
from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[str]] = None

class ChatResponse(BaseModel):
    # Primary answer
    answer: str
    sources: List[str]
    retrieval_path: List[Dict[str, Any]]
    transformed_query: Optional[str] = None
    
    # Alternative answer (when dual mode is active)
    has_alternative: bool = False
    alternative_answer: Optional[str] = None
    alternative_sources: Optional[List[str]] = None
    alternative_retrieval_path: Optional[List[Dict[str, Any]]] = None
    alternative_type: Optional[str] = None  # "retrieval" or "gold"
    
    # Metadata
    answer_mode: str = "retrieval_only"  # "gold_only", "retrieval_only", or "dual"
    direct_gold_match: bool = False
    gold_similarity: Optional[float] = None