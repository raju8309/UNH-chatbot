from pathlib import Path
from typing import Any
import os

try:
    from config.settings import EMBED_MODEL_NAME as _CFG_EMBED_NAME
except Exception:
    _CFG_EMBED_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

from sentence_transformers import SentenceTransformer
from transformers import pipeline

# global model instances
embed_model: SentenceTransformer = None
qa_pipeline: Any = None
_loaded_embed_model_name: str = None

def initialize_models(fine_tuned : bool = True) -> None:
    global embed_model, qa_pipeline
    
    # choose embedding model name from settings or ENV fallback
    model_name = _CFG_EMBED_NAME or os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    embed_model = SentenceTransformer(model_name)
    globals()["_loaded_embed_model_name"] = model_name
    print(f"Embedding model loaded: {model_name}")
    
    # load trained model if available, otherwise use default
    trained_path = Path(__file__).parent.parent / "train" / "models" / "flan-t5-small-finetuned"
    
    if fine_tuned and trained_path.exists() and (trained_path / "config.json").exists():
        model_name = str(trained_path)
        print(f"Using fine-tuned model: {trained_path}")
    else:
        model_name = "google/flan-t5-small"
        print(f"Using default model: {model_name}")
    
    qa_pipeline = pipeline(
        "text2text-generation",
        model=model_name,
        device=-1,  # CPU
    )
    print("QA pipeline loaded")

def get_embed_model() -> SentenceTransformer:
    """
    Returns the cached embedding model. If initialize_models() wasn't called yet,
    initialize on the fly using the configured name.
    """
    global embed_model, _loaded_embed_model_name
    if embed_model is None:
        # lazy init fallback (keeps behavior robust in tests/scripts)
        model_name = _CFG_EMBED_NAME or os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
        embed_model = SentenceTransformer(model_name)
        _loaded_embed_model_name = model_name
        print(f"[lazy-init] Embedding model loaded: {model_name}")
    return embed_model

def get_qa_pipeline() -> Any:
    if qa_pipeline is None:
        raise RuntimeError("QA pipeline not initialized. Call initialize_models() first.")
    return qa_pipeline