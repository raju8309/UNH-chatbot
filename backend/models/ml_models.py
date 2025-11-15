from pathlib import Path
from typing import Any
from config.settings import get_config
from typing import Any, List
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
    
    # Check config for fine-tuned model setting
    cfg = get_config()
    fine_tuned = cfg.get("performance", {}).get("use_finetuned_model", fine_tuned)
    # choose embedding model name from settings or ENV fallback
    model_name = _CFG_EMBED_NAME or os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    
    embed_model = SentenceTransformer(model_name, device='cpu')
    globals()["_loaded_embed_model_name"] = model_name
    print(f"Embedding model loaded: {model_name} (CPU)")
    
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
        embed_model = SentenceTransformer(model_name, device='cpu')
        _loaded_embed_model_name = model_name
        print(f"[lazy-init] Embedding model loaded: {model_name} (CPU)")
    return embed_model

def get_qa_pipeline() -> Any:
    if qa_pipeline is None:
        raise RuntimeError("QA pipeline not initialized. Call initialize_models() first.")
    return qa_pipeline

def _ensure_qa_pipeline():
    """
    Ensure the text2text pipeline is available. If initialize_models() was not
    called elsewhere (e.g., app startup), try to initialize it now with defaults.
    """
    global qa_pipeline
    if qa_pipeline is None:
        # initialize with defaults; this mirrors initialize_models(False) semantics
        initialize_models(fine_tuned=True)
    return qa_pipeline

def call_model(prompt: str) -> str:
    """
    Generic callable used by services/query_transform_service.py for short rewrites.
    It uses the existing text2text 'qa_pipeline' (FLAN-T5 small or your finetuned variant).
    Returns the generated text as a stripped string.
    """
    pipe = _ensure_qa_pipeline()
    try:
        # Keep decoding deterministic for stability; tweak as you like.
        outputs = pipe(
            prompt,
            max_new_tokens=64,
            do_sample=False,
            num_return_sequences=1,
            clean_up_tokenization_spaces=True
        )
        if isinstance(outputs, list) and outputs:
            text = outputs[0].get("generated_text") or outputs[0].get("summary_text") or str(outputs[0])
        else:
            # Fallback to string conversion if pipeline returns a dict
            text = str(outputs)
        return (text or "").strip()
    except Exception as e:
        # Surface a concise message; the caller has its own fallback path.
        raise RuntimeError(f"call_model generation failed: {e}") from e

def generate_text(prompt: str) -> str:
    """Alias to call_model for compatibility."""
    return call_model(prompt)

def get_text_embedding(text: str) -> List[float]:
    """Return a single text embedding as a list of floats.

    This reuses the global SentenceTransformer instance so that other
    services (e.g. query_transform_service) can compute semantic
    similarity between the original and rewritten queries.
    """
    model = get_embed_model()
    # encode returns a numpy array when convert_to_numpy=True; we convert
    # to a plain Python list for easier downstream use.
    emb = model.encode([text], convert_to_numpy=True)[0]
    return emb.tolist()