import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from pydantic import BaseModel

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
import csv
import threading
from datetime import datetime

# --- CSV log ---
CHAT_LOG_PATH = "chat_logs.csv"
_LOG_LOCK = threading.Lock()
if not os.path.isfile(CHAT_LOG_PATH):
    with open(CHAT_LOG_PATH, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["timestamp", "question", "answer", "sources_json"])

# --- FastAPI ---
app = FastAPI()

# If you mount the dashboard router elsewhere, keep it; otherwise comment out
try:
    from dashboard import router as dashboard_router
    app.include_router(dashboard_router)
except Exception:
    pass

PUBLIC_URL = os.getenv("PUBLIC_URL", "http://localhost:8003/")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[PUBLIC_URL],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

# --- Index state ---
chunks_embeddings = None
chunk_texts = []
chunk_sources = []

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "scraper"

# Models
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-small", device=-1)

def _base_doc_id(url: str) -> str:
    if not url:
        return "catalog"
    p = urlparse(url)
    name = (Path(p.path).name or "").strip("/") or (Path(p.path).parts[-1] if p.path else "")
    slug = (name or "catalog").replace(".html", "").replace(".htm", "") or "catalog"
    return slug

def load_json_file(path):
    """Append chunks from a UNH catalog JSON into the global index."""
    global chunks_embeddings, chunk_texts, chunk_sources
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_texts, new_sources = [], []

    def recurse_sections(sections, parent_title=""):
        for sec in sections:
            title = sec.get("title", "")
            full_title = f"{parent_title} > {title}" if parent_title else title

            for t in sec.get("text", []):
                new_texts.append(t)
                new_sources.append({"title": full_title, "url": sec.get("page_url", "")})

            for link in sec.get("links", []):
                label, url = link.get("label"), link.get("url")
                if label and url:
                    new_texts.append(f"Courses: {label}")
                    new_sources.append({"title": label, "url": url})

            if "subsections" in sec:
                recurse_sections(sec["subsections"], parent_title=full_title)

    recurse_sections(data.get("sections", []))

    if new_texts:
        if chunks_embeddings is None:
            chunks_embeddings = embed_model.encode(new_texts, convert_to_numpy=True)
            chunk_texts[:] = new_texts
            chunk_sources[:] = new_sources
        else:
            new_embeds = embed_model.encode(new_texts, convert_to_numpy=True)
            chunks_embeddings = np.vstack([chunks_embeddings, new_embeds])
            chunk_texts.extend(new_texts)
            chunk_sources.extend(new_sources)
        print(f"Loaded {len(new_texts)} chunks from {path}")
    else:
        print(f"WARNING: no text found in {path}")

# ---------- Public helpers for tests ----------
def build_index_from_jsons(paths: list[str]):
    """Idempotent-ish builder that appends catalog JSONs to the index."""
    for p in paths:
        p = Path(p)
        if p.exists():
            load_json_file(str(p))
        else:
            print(f"WARNING: {p} not found, skipping.")

def get_top_chunks(question, top_k=3):
    if chunks_embeddings is None or len(chunks_embeddings) == 0:
        return []
    qv = embed_model.encode([question], convert_to_numpy=True)[0]
    scores = np.dot(chunks_embeddings, qv) / (
        np.linalg.norm(chunks_embeddings, axis=1) * np.linalg.norm(qv) + 1e-10
    )
    idxs = scores.argsort()[-top_k:][::-1]
    results = []
    for i in idxs:
        text = chunk_texts[i]
        src = chunk_sources[i]
        base = _base_doc_id(src.get("url", ""))
        cid = f"{base}#{i}"
        results.append((text, src, cid))
    return results

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

def _answer_question(question):
    triples = get_top_chunks(question)
    top_chunks = [(t, s) for (t, s, _) in triples]
    context = " ".join([t for t, _ in top_chunks])
    prompt = (
        "Answer the question ONLY using the provided context. "
        "If the answer cannot be found, say you don't know. "
        "If the context does not mention the degree, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    result = qa_pipeline(prompt, max_new_tokens=128)
    answer = result[0]["generated_text"].strip()
    enriched_sources = _wrap_sources_with_text_fragments(top_chunks, question)

    seen, citation_lines = set(), []
    for src in enriched_sources:
        key = (src.get("title"), src.get("url"))
        if key in seen:
            continue
        seen.add(key)
        line = f"- {src.get('title','Source')}"
        if src.get("url"):
            line += f" ({src['url']})"
        citation_lines.append(line)
    retrieved_ids = [cid for (_, _, cid) in triples]
    return answer, citation_lines, retrieved_ids

@lru_cache(maxsize=128)
def cached_answer_tuple(question_str):
    return _answer_question(question_str)

# Public function for tests
def answer_with_sources(question: str, top_k: int = 3):
    triples = get_top_chunks(question, top_k=top_k)
    if not triples:
        return "I don't know.", [], []
    # re-run with desired k
    top_chunks = [(t, s) for (t, s, _) in triples]
    context = " ".join([t for t, _ in top_chunks])
    prompt = (
        "Answer the question ONLY using the provided context. "
        "If the answer cannot be found, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    result = qa_pipeline(prompt, max_new_tokens=128)
    answer = result[0]["generated_text"].strip()
    sources = _wrap_sources_with_text_fragments(top_chunks, question)
    retrieved_ids = [cid for (_, _, cid) in triples]
    return answer, [f"- {s.get('title')}" + (f" ({s.get('url')})" if s.get('url') else "") for s in sources], retrieved_ids

# ---------- FastAPI models & routes ----------
class ChatRequest(BaseModel):
    message: str
    history: list[str] | None = None

class ChatResponse(BaseModel):
    answer: str
    sources: list[str] = []

def log_chat_to_csv(question, answer, sources):
    ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    row = [ts, question, answer, json.dumps(sources, ensure_ascii=False)]
    with _LOG_LOCK:
        with open(CHAT_LOG_PATH, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)

@app.post("/reset")
async def reset_chat():
    return {"ok": True}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    msg = request.message
    if isinstance(msg, list):
        msg = " ".join(msg)
    answer, sources, _ = answer_with_sources(msg, top_k=3)
    log_chat_to_csv(msg, answer, sources)
    return ChatResponse(answer=answer, sources=sources)

# Mount static UI if present
frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../frontend/out'))
if os.path.isdir(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
    print("Mounted frontend from:", frontend_path)

# --- IMPORTANT: don't auto-build index or run server when imported by tests ---
def _default_scrape_paths():
    return [str(DATA_DIR / "unh_catalog.json")]

if __name__ == "__main__" and os.getenv("RUN_API", "1") == "1":
    # Build index only when serving API
    build_index_from_jsons(_default_scrape_paths())
    uvicorn.run("main:app", host="0.0.0.0", port=8003, reload=True)
