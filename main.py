import os
import csv
import json
import threading
from datetime import datetime
from functools import lru_cache
from typing import List, Dict, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from pydantic import BaseModel

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

# -----------------------------
# App & CORS
# -----------------------------
app = FastAPI()

PUBLIC_URL = os.getenv("PUBLIC_URL", "http://localhost:8003/t3/")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[PUBLIC_URL],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Chat log setup (CSV)
# -----------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DEFAULT_LOG_PATH = os.path.join(BASE_DIR, "chat_logs.csv")  # <- single file next to main.py
CHAT_LOG_PATH = os.getenv("CHAT_LOG_PATH", DEFAULT_LOG_PATH)

# Ensure folder exists (the file’s parent directory)
os.makedirs(os.path.dirname(CHAT_LOG_PATH) or ".", exist_ok=True)

# Write header once if file doesn’t exist
if not os.path.isfile(CHAT_LOG_PATH):
    with open(CHAT_LOG_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "question", "answer", "sources_json"])

# Simple lock so concurrent requests don’t interleave writes
_LOG_LOCK = threading.Lock()


def log_chat_to_csv(question: str, answer: str, sources: List[Dict]):
    """Append one chat row to CSV (thread-safe)."""
    ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    sources_json = json.dumps(sources, ensure_ascii=False)
    with _LOG_LOCK:
        with open(CHAT_LOG_PATH, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ts, question, answer, sources_json])

# -----------------------------
# Retrieval state
# -----------------------------
chunks_embeddings: np.ndarray | None = None
chunk_texts: List[str] = []
chunk_sources: List[Dict] = []

# Local embedding model
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Local QA model (Flan-T5 Small)
qa_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    device=-1,  # CPU
)

def load_json_file(path: str):
    """Load a JSON file into global embeddings/texts/sources."""
    global chunks_embeddings, chunk_texts, chunk_sources

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_texts: List[str] = []
    new_sources: List[Dict] = []

    def recurse_sections(sections, parent_title=""):
        for sec in sections:
            title = sec.get("title", "")
            full_title = f"{parent_title} > {title}" if parent_title else title

            # add plain text
            for t in sec.get("text", []):
                new_texts.append(t)
                new_sources.append({"title": full_title, "url": data.get("url", "")})

            # add links (label + URL)
            for link in sec.get("links", []):
                label = link.get("label")
                url = link.get("url")
                if label and url:
                    new_texts.append(f"Courses: {label}")
                    new_sources.append({"title": label, "url": url})

            if "subsections" in sec:
                recurse_sections(sec["subsections"], parent_title=full_title)

    recurse_sections(data.get("sections", []))

    if new_texts:
        if chunks_embeddings is None:
            chunks_embeddings = embed_model.encode(new_texts, convert_to_numpy=True)
            chunk_texts.extend(new_texts)
            chunk_sources.extend(new_sources)
        else:
            new_embeds = embed_model.encode(new_texts, convert_to_numpy=True)
            chunks_embeddings = np.vstack([chunks_embeddings, new_embeds])
            chunk_texts.extend(new_texts)
            chunk_sources.extend(new_sources)

        print(f"Loaded {len(new_texts)} chunks from {path}")
    else:
        print(f"WARNING: no text found in {path}")

def get_top_chunks(question: str, top_k: int = 3) -> List[Tuple[str, Dict]]:
    if chunks_embeddings is None or len(chunks_embeddings) == 0:
        return []
    question_vec = embed_model.encode([question], convert_to_numpy=True)[0]
    denom = (np.linalg.norm(chunks_embeddings, axis=1) * np.linalg.norm(question_vec) + 1e-10)
    scores = np.dot(chunks_embeddings, question_vec) / denom
    top_indices = scores.argsort()[-top_k:][::-1]
    return [(chunk_texts[i], chunk_sources[i]) for i in top_indices]

def _answer_question(question: str) -> Tuple[str, List[Dict]]:
    top_chunks = get_top_chunks(question)
    context = " ".join([text for text, _ in top_chunks])

    prompt = (
        "Answer the question ONLY using the provided context. "
        "If the answer cannot be found, say you don't know. "
        "If the context does not mention the degree, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

    try:
        result = qa_pipeline(prompt, max_new_tokens=128)
        answer = result[0]["generated_text"].strip()

        # build unique sources (preserve order)
        seen = set()
        sources: List[Dict] = []
        for _, src in top_chunks:
            key = (src.get("title"), src.get("url"))
            if key not in seen:
                seen.add(key)
                sources.append({"title": src.get("title", ""), "url": src.get("url", "")})

        return answer, sources
    except Exception as e:
        return f"ERROR running local model: {e}", []

@lru_cache(maxsize=128)
def cached_answer_tuple(question_str: str) -> Tuple[str, List[Dict]]:
    return _answer_question(question_str)

# -----------------------------
# Schemas & Route
# -----------------------------
class ChatRequest(BaseModel):
    message: str
    history: List[str] | None = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict] = []

@app.post("/t3/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    message = request.message
    if isinstance(message, list):
        message = " ".join(message)

    answer, sources = cached_answer_tuple(message)

    # Log exchange
    log_chat_to_csv(message, answer, sources)

    return ChatResponse(answer=answer, sources=sources)

# -----------------------------
# Frontend mount (optional)
# -----------------------------
frontend_path = os.path.join(BASE_DIR, "frontend", "out")
if os.path.isdir(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
    print("Mounted frontend from:", frontend_path)

# -----------------------------
# Load data on startup
# -----------------------------
def _try_load_data():
    # Try common locations for the scraped JSON files
    candidates = [
        os.path.join(BASE_DIR, "scrape", "course_descriptions.json"),
        os.path.join(BASE_DIR, "scrape", "degree_requirements.json"),
        os.path.join(BASE_DIR, "scraper", "course_descriptions.json"),
        os.path.join(BASE_DIR, "scraper", "degree_requirements.json"),
    ]

    # Try in pairs (course_descriptions + degree_requirements)
    loaded_any = False
    for i in range(0, len(candidates), 2):
        a, b = candidates[i], candidates[i + 1]
        if os.path.isfile(a) and os.path.isfile(b):
            try:
                load_json_file(a)
                load_json_file(b)
                loaded_any = True
                break
            except Exception as e:
                print(f"Failed loading {a} or {b}: {e}")

    if not loaded_any:
        print("WARNING: No scrape JSON files were loaded. Retrieval will return empty context.")

_try_load_data()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8003, reload=True)
