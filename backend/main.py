import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from pydantic import BaseModel

import re
import json
from functools import lru_cache
from text_fragments import build_text_fragment_url, choose_snippet, is_synthetic_label
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from fastapi.staticfiles import StaticFiles
import os

import pickle
import csv
import threading
from datetime import datetime

# Import dashboard router
from dashboard import router as dashboard_router

# --- CSV logging ---
CHAT_LOG_PATH = "chat_logs.csv"
_LOG_LOCK = threading.Lock()

# FastAPI backend app
app = FastAPI()

# Global variables
chunks_embeddings = None
chunk_texts = []
chunk_sources = []

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "scraper"
CACHE_PATH = DATA_DIR / "chunks_cache.pkl"

# Embeddings model
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# QA pipeline
qa_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    device=-1
)

# cache helpers
def save_chunks_cache():
    global chunk_texts, chunk_sources, chunks_embeddings
    with open(CACHE_PATH, "wb") as f:
        pickle.dump({
            "texts": chunk_texts,
            "sources": chunk_sources,
            "embeddings": chunks_embeddings
        }, f)
    print(f"saved {len(chunk_texts)} chunks to cache")

# load cache for improved startup speed
def load_chunks_cache():
    global chunk_texts, chunk_sources, chunks_embeddings
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "rb") as f:
            data = pickle.load(f)
        chunk_texts = data["texts"]
        chunk_sources = data["sources"]
        chunks_embeddings = data["embeddings"]
        print(f"Loaded {len(chunk_texts)} chunks from cache.")
        return True
    return False

# --- JSON loader ---
def load_json_file(path):
    """Load JSON and append chunks and embeddings."""
    global chunks_embeddings, chunk_texts, chunk_sources

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_texts = []
    new_sources = []

    def recurse_sections(sections, parent_title=""):
        for sec in sections:
            title = sec.get("title", "")
            full_title = f"{parent_title} > {title}" if parent_title else title

            for t in sec.get("text", []):
                new_texts.append(t)
                new_sources.append({
                    "title": full_title,
                    "url": sec.get("page_url", "")
                })

            for link in sec.get("links", []):
                label = link.get("label")
                url = link.get("url")
                if label and url:
                    new_texts.append(f"Courses: {label}")
                    new_sources.append({
                        "title": label,
                        "url": url
                    })

            if "subsections" in sec:
                recurse_sections(sec["subsections"], parent_title=full_title)

    recurse_sections(data.get("sections", []))

    if new_texts:
        new_embeds = embed_model.encode(new_texts, convert_to_numpy=True)
        if chunks_embeddings is None:
            chunks_embeddings = new_embeds
        else:
            chunks_embeddings = np.vstack([chunks_embeddings, new_embeds])
        chunk_texts.extend(new_texts)
        chunk_sources.extend(new_sources)
        print(f"Loaded {len(new_texts)} chunks from {path}")
    else:
        print(f"WARNING: no text found in {path}")

def load_catalog(path: str):
    p = Path(path)
    if p.exists():
        load_json_file(str(p))
    else:
        print(f"WARNING: {p} not found, skipping.")

# retrieval utilities
def get_top_chunks(question, top_k=3):
    if chunks_embeddings is None or len(chunks_embeddings) == 0:
        return []
    question_vec = embed_model.encode([question], convert_to_numpy=True)[0]
    scores = np.dot(chunks_embeddings, question_vec) / (
        np.linalg.norm(chunks_embeddings, axis=1) * np.linalg.norm(question_vec) + 1e-10
    )
    top_indices = scores.argsort()[-top_k:][::-1]
    return [(chunk_texts[i], chunk_sources[i]) for i in top_indices]

def _wrap_sources_with_text_fragments(sources_with_passages, question: str):
    wrapped = []
    for passage, src in sources_with_passages:
        url = src.get("url", "")
        if not url or is_synthetic_label(passage):
            wrapped.append({**src, "url": url})
            continue
        snippet = choose_snippet(passage, hint=question, max_chars=160)
        if snippet:
            frag_url = build_text_fragment_url(url, text=snippet)
            wrapped.append({**src, "url": frag_url})
        else:
            wrapped.append({**src, "url": url})
    return wrapped

# --- Answer ---
def _answer_question(question):
    top_chunks = get_top_chunks(question)
    context = " ".join([text for text, _ in top_chunks])

    try:
        # Short answer
        short_prompt = (
            "Answer the question using ONLY the provided context. "
            "Provide a short, factual answer first (like a number, date, or 'Yes/No'). "
            "If the answer cannot be found in the context, say you don't know.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        short_result = qa_pipeline(short_prompt, max_new_tokens=32)
        short_answer = short_result[0]["generated_text"].strip()

        # Long answer
        long_prompt = (
            "Provide a brief, natural, and informative explanation in 2–3 complete sentences. "
            "Use ONLY the provided context.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        long_result = qa_pipeline(long_prompt, max_new_tokens=128)
        long_answer = long_result[0]["generated_text"].strip()

        # Combine answers
        if short_answer and long_answer:
            if long_answer.lower().startswith(short_answer):
                answer = long_answer
            else:
                answer = f"{short_answer}. {long_answer}"
        else:
            answer = short_answer or long_answer

        # Limit to 3 sentences
        def shorten_to_sentences(text, max_sentences=3):
            sentences = re.split(r'(?<=[.!?]) +', text)
            return " ".join(sentences[:max_sentences]).strip()

        answer = shorten_to_sentences(answer, max_sentences=3)

        # Build citations
        enriched_sources = _wrap_sources_with_text_fragments(top_chunks, question)
        seen = set()
        citation_lines = []
        for src in enriched_sources:
            key = (src.get("title"), src.get("url"))
            if key in seen:
                continue
            seen.add(key)
            line = f"- {src.get('title','Source')}"
            if src.get("url"):
                line += f" ({src['url']})"
            citation_lines.append(line)

        return answer, citation_lines

    except Exception as e:
        return f"ERROR running local model: {e}", []

# --- Cached wrapper ---
@lru_cache(maxsize=128)
def cached_answer_tuple(question_str):
    return _answer_question(question_str)

# small test helper: answer + retrieved_ids
def _base_doc_id(url: str) -> str:
    if not url:
        return "catalog"
    p = urlparse(url)
    name = (Path(p.path).name or "").rstrip("/")
    if not name and p.path:
        name = Path(p.path).parts[-1]
    slug = (name or "catalog").replace(".html", "").replace(".htm", "") or "catalog"
    return slug

def answer_with_sources(question: str, top_k: int = 3):
    """
    Test helper that returns (answer_text_only, sources_list, retrieved_ids).
    NOTE: Answer text contains NO embedded sources (tests should write only the answer).
    """
    if chunks_embeddings is None or len(chunks_embeddings) == 0:
        return "I don't know.", [], []

    # recompute top indices (leave get_top_chunks unchanged)
    qv = embed_model.encode([question], convert_to_numpy=True)[0]
    scores = np.dot(chunks_embeddings, qv) / (
        np.linalg.norm(chunks_embeddings, axis=1) * np.linalg.norm(qv) + 1e-10
    )
    idxs = scores.argsort()[-top_k:][::-1]
    top_pairs = [(chunk_texts[i], chunk_sources[i]) for i in idxs]
    context = " ".join([t for t, _ in top_pairs])

    prompt = (
        "Answer the question ONLY using the provided context. "
        "If the answer cannot be found, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    result = qa_pipeline(prompt, max_new_tokens=128)
    answer = result[0]["generated_text"].strip()

    # build retrieved_ids in '<docid>#<chunk_index>' format
    retrieved_ids = []
    for i in idxs:
        src = chunk_sources[i]
        base = _base_doc_id(src.get("url", ""))
        retrieved_ids.append(f"{base}#{i}")

    # separate sources list (test runner can ignore)
    sources_list = []
    for _, src in top_pairs:
        line = f"- {src.get('title','Source')}"
        if src.get("url"):
            line += f" ({src['url']})"
        sources_list.append(line)

    return answer, sources_list, retrieved_ids

# --- FastAPI models ---
class ChatRequest(BaseModel):
    message: str
    history: list[str] = None

class ChatResponse(BaseModel):
    answer: str
    sources: list[str] = []

# --- CSV logging ---
def log_chat_to_csv(question, answer, sources):
    ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    row = [ts, question, answer, json.dumps(sources, ensure_ascii=False)]
    with _LOG_LOCK:
        with open(CHAT_LOG_PATH, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)

# --- FastAPI endpoints ---
@app.post("/reset")
async def reset_chat():
    pass

@app.post("/chat", response_model=ChatResponse)
async def answer_question(request: ChatRequest):
    message = request.message
    if isinstance(message, list):
        message = " ".join(message)
    answer, sources = cached_answer_tuple(message)
    log_chat_to_csv(message, answer, sources)
    return ChatResponse(answer=answer, sources=sources)

if __name__ == "__main__":
    # Load catalog
    load_catalog(DATA_DIR / "unh_catalog.json")
    # create file with header if not exists
    if not os.path.isfile(CHAT_LOG_PATH):
        with open(CHAT_LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "question", "answer", "sources_json"])
    # Include dashboard router
    app.include_router(dashboard_router)
    # Allow CORS for frontend
    PUBLIC_URL = os.getenv("PUBLIC_URL", "http://localhost:8003/")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[PUBLIC_URL],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"]
    )
    # Mount static files at root after all API routes
    frontend_path = f"{BASE_DIR}/frontend/out"
    if os.path.isdir(frontend_path):
        app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
        print("Mounted frontend from:", frontend_path)
    # Run server
    uvicorn.run(app, host="0.0.0.0", port=8003, reload=False)
