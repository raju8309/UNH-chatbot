import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from pydantic import BaseModel

import json
from functools import lru_cache
from text_fragments import build_text_fragment_url, choose_snippet, is_synthetic_label
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from fastapi.staticfiles import StaticFiles
import os

# --- CSV logging (added) ---
import csv
import threading
from datetime import datetime

CHAT_LOG_PATH = "chat_logs.csv"
_LOG_LOCK = threading.Lock()

# create file with header if not exists
if not os.path.isfile(CHAT_LOG_PATH):
    with open(CHAT_LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "question", "answer", "sources_json"])
# --- end CSV logging ---

# FastAPI backend app
app = FastAPI()

# Allow CORS for frontend
PUBLIC_URL = os.getenv("PUBLIC_URL", "http://localhost:8003/")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[PUBLIC_URL],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# for chunking text
chunks_embeddings = None
chunk_texts = []
chunk_sources = []

# paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "scraper"

# embeddings model (local + small)
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# load local Flan-T5 Small for queries
qa_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    device=-1,  # CPU currently better performance on GPU (0)
)

def load_json_file(path):
    """Load a JSON file into global embeddings/texts/sources."""
    global chunks_embeddings, chunk_texts, chunk_sources

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_texts = []
    new_sources = []

    def recurse_sections(sections, parent_title=""):
        for sec in sections:
            title = sec.get("title", "")
            full_title = f"{parent_title} > {title}" if parent_title else title

            # add plain text
            for t in sec.get("text", []):
                new_texts.append(t)
                new_sources.append({
                    "title": full_title,
                    "url": data.get("url", "")
                })

            # add links (label + URL)
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

    # append new chunks to existing ones
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
    """
    Input: list of tuples (passage_text, source_dict)
    Output: list of dicts like source_dict but with url replaced by a text-fragment URL
    """
    wrapped = []
    for passage, src in sources_with_passages:
        url = src.get("url", "")
        # Some chunks are synthetic labels (e.g., "Courses: …"); don't build fragments for those
        if not url or is_synthetic_label(passage):
            wrapped.append({**src, "url": url})
            continue

        # Choose a compact snippet (tries to align to the question)
        snippet = choose_snippet(passage, hint=question, max_chars=160)

        if snippet:
            frag_url = build_text_fragment_url(url, text=snippet)
            wrapped.append({**src, "url": frag_url})
        else:
            wrapped.append({**src, "url": url})
    return wrapped

# core answer function (not cached directly)
def _answer_question(question):
    top_chunks = get_top_chunks(question)
    context = " ".join([text for text, _ in top_chunks])

    prompt = (
        "Answer the question ONLY using the provided context. "
        "If the answer cannot be found, say you don't know. "
        "If the context does not mention the degree, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

    try:
        result = qa_pipeline(prompt, max_new_tokens=128)  # limit tokens
        answer = result[0]["generated_text"].strip()

        # Build citation links WITH text fragments, one per unique (title,url)
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

        citations = "\n".join(citation_lines)
        return f"{answer}\n\nSources:\n{citations}"
    except Exception as e:
        return f"ERROR running local model: {e}"

# cached wrapper for answers
@lru_cache(maxsize=128)
def cached_answer_tuple(question_str):
    return _answer_question(question_str)

# FastAPI request model
class ChatRequest(BaseModel):
    message: str
    history: list[str] = None

# FastAPI response model
class ChatResponse(BaseModel):
    answer: str
    sources: list[dict] = []

# --- CSV logging helper (added) ---
def log_chat_to_csv(question, answer, sources):
    ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    row = [ts, question, answer, json.dumps(sources, ensure_ascii=False)]
    with _LOG_LOCK:
        with open(CHAT_LOG_PATH, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
# --- end helper ---

# FastAPI chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def answer_question(request: ChatRequest):
    message = request.message
    if isinstance(message, list):
        message = " ".join(message)
    answer, sources = cached_answer_tuple(message)

    # --- log to CSV (added) ---
    log_chat_to_csv(message, answer, sources)

    return ChatResponse(answer=answer, sources=sources)

# Mount static files at root after all API routes
frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../frontend/out'))
if os.path.isdir(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
    print("Mounted frontend from:", frontend_path)


# Load data files
load_json_file("scraper/course_descriptions.json")
load_json_file("scraper/degree_requirements.json")


if __name__ == "__main__":
    # Load scraped JSONs from the scraper/ folder
    filenames = [
        "course_descriptions.json",
        "degree_requirements.json",
        "academic_standards.json",
        "graduation.json",
        "graduation_grading.json",
    ]
    for name in filenames:
        path = DATA_DIR / name
        if path.exists():
            load_json_file(str(path))
        else:
            print(f"WARNING: {path} not found, skipping.")

    demo.launch()
