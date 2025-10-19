from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer
from transformers import pipeline

# global model instances
embed_model: SentenceTransformer = None
qa_pipeline: Any = None

def initialize_models() -> None:
    global embed_model, qa_pipeline
    
    # initialize embedding model
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("Embedding model loaded")
    
    # load trained model if available, otherwise use default
    trained_path = Path(__file__).parent.parent / "train" / "models" / "flan-t5-small-finetuned"
    
    if trained_path.exists() and (trained_path / "config.json").exists():
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
    if embed_model is None:
        raise RuntimeError("Embedding model not initialized. Call initialize_models() first.")
    return embed_model

def get_qa_pipeline() -> Any:
    if qa_pipeline is None:
        raise RuntimeError("QA pipeline not initialized. Call initialize_models() first.")
    return qa_pipeline