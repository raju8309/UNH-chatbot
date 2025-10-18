from typing import Any, Dict, List, Optional
from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[str]] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    retrieval_path: List[Dict[str, Any]]