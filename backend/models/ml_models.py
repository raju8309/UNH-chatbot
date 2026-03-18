from pathlib import Path
from typing import Any, List
from config.settings import get_config
import os

try:
    from config.settings import EMBED_MODEL_NAME as _CFG_EMBED_NAME
except Exception:
    _CFG_EMBED_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# global model instances
embed_model: SentenceTransformer = None
qa_pipeline: Any = None
_qa_tokenizer = None
_qa_model = None
_loaded_embed_model_name: str = None


def initialize_models(fine_tuned: bool = True) -> None:
    global embed_model, qa_pipeline, _qa_tokenizer, _qa_model

    cfg = get_config()
    fine_tuned = cfg.get("performance", {}).get("use_finetuned_model", fine_tuned)

    model_name = _CFG_EMBED_NAME or os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    embed_model = SentenceTransformer(model_name, device='cpu')
    globals()["_loaded_embed_model_name"] = model_name
    print(f"Embedding model loaded: {model_name} (CPU)")

    trained_path = Path(__file__).parent.parent / "train" / "models" / "flan-t5-small-finetuned"

    if fine_tuned and trained_path.exists() and (trained_path / "config.json").exists():
        model_name = str(trained_path)
        print(f"Using fine-tuned model: {trained_path}")
    else:
        model_name = "google/flan-t5-small"
        print(f"Using default model: {model_name}")

    # ✅ FIX: Skip pipeline entirely, use model directly
    _qa_tokenizer = AutoTokenizer.from_pretrained(model_name)
    _qa_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    _qa_model.eval()

    # Create a simple callable that mimics pipeline output
    def _pipeline_fn(prompt, max_new_tokens=64, do_sample=False, num_return_sequences=1, **kwargs):
        inputs = _qa_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = _qa_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                num_return_sequences=num_return_sequences,
            )
        decoded = _qa_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return [{"generated_text": decoded}]

    qa_pipeline = _pipeline_fn
    print("QA pipeline loaded (direct model inference)")


def get_embed_model() -> SentenceTransformer:
    global embed_model, _loaded_embed_model_name
    if embed_model is None:
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
    global qa_pipeline
    if qa_pipeline is None:
        initialize_models(fine_tuned=True)
    return qa_pipeline


def call_model(prompt: str) -> str:
    pipe = _ensure_qa_pipeline()
    try:
        outputs = pipe(
            prompt,
            max_new_tokens=64,
            do_sample=False,
            num_return_sequences=1,
        )
        if isinstance(outputs, list) and outputs:
            text = outputs[0].get("generated_text") or outputs[0].get("summary_text") or str(outputs[0])
        else:
            text = str(outputs)
        return (text or "").strip()
    except Exception as e:
        raise RuntimeError(f"call_model generation failed: {e}") from e


def generate_text(prompt: str) -> str:
    return call_model(prompt)


def get_text_embedding(text: str) -> List[float]:
    model = get_embed_model()
    emb = model.encode([text], convert_to_numpy=True)[0]
    return emb.tolist()