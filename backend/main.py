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

import csv
import threading
from datetime import datetime

CHAT_LOG_PATH = "chat_logs.csv"
_LOG_LOCK = threading.Lock()

if not os.path.isfile(CHAT_LOG_PATH):
    with open(CHAT_LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "question", "answer", "sources_json"])

app = FastAPI()
PUBLIC_URL = os.getenv("PUBLIC_URL", "http://localhost:8003")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[PUBLIC_URL],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

chunks_embeddings = None
chunk_texts = []
chunk_sources = []

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "scraper"

embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
qa_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    device=-1,
)

# load json and build chunks
def load_json_file(path):
    global chunks_embeddings, chunk_texts, chunk_sources

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_texts = []
    new_sources = []

    def recurse_sections(sections, parent_title="", parent_url=""):
        for sec in sections:
            title = sec.get("title", "")
            full_title = f"{parent_title} > {title}" if parent_title else title
            section_url = sec.get("url", parent_url)

            # add plain text
            for t in sec.get("text", []):
                new_texts.append(t)
                new_sources.append({
                    "title": full_title,
                    "url": section_url
                })

            # add links
            for link in sec.get("links", []):
                label = link.get("label")
                url = link.get("url")
                if label and url:
                    new_texts.append(f"Courses: {label}")
                    new_sources.append({
                        "title": label,
                        "url": url
                    })

            # recurse subsections
            if "subsections" in sec:
                recurse_sections(sec["subsections"], parent_title=full_title, parent_url=section_url)

    if isinstance(data, dict):
        recurse_sections(data.get("sections", []), parent_url=data.get("url", ""))
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                recurse_sections(item.get("sections", []), parent_url=item.get("url", ""))

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

# get top chunks acquire proper urls
def get_top_chunks(question, top_k=3, threshold=0.65):
    if chunks_embeddings is None or len(chunks_embeddings) == 0:
        return []

    question_vec = embed_model.encode([question], convert_to_numpy=True)[0]
    scores = np.dot(chunks_embeddings, question_vec) / (
        np.linalg.norm(chunks_embeddings, axis=1) * np.linalg.norm(question_vec) + 1e-10
    )

    # boost chunks with URLs slightly
    for i, src in enumerate(chunk_sources):
        if src.get("url"):
            scores[i] += 0.05

    # select chunks, keep all with URLs above 0 filter others by threshold
    selected_indices = [
        i for i, s in enumerate(scores)
        if (chunk_sources[i].get("url") or s >= threshold)
    ]

    if not selected_indices:
        selected_indices = [scores.argmax()]

    # sort by score and limit top_k
    selected_indices = sorted(selected_indices, key=lambda i: scores[i], reverse=True)[:top_k]
    return [(chunk_texts[i], chunk_sources[i]) for i in selected_indices]

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
    top_chunks = get_top_chunks(question)
    context = " ".join([text for text, _ in top_chunks])

    prompt = (
        "Answer the question ONLY using the provided context. "
        "If the answer cannot be found, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

    try:
        result = qa_pipeline(prompt, max_new_tokens=128)
        answer = result[0]["generated_text"].strip()

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

@lru_cache(maxsize=128)
def cached_answer_tuple(question_str):
    return _answer_question(question_str)

class ChatRequest(BaseModel):
    message: str
    history: list[str] = None

class ChatResponse(BaseModel):
    answer: str
    sources: list[str] = []

def log_chat_to_csv(question, answer, sources):
    ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    row = [ts, question, answer, json.dumps(sources, ensure_ascii=False)]
    with _LOG_LOCK:
        with open(CHAT_LOG_PATH, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)

@app.post("/chat", response_model=ChatResponse)
async def answer_question(request: ChatRequest):
    message = request.message
    if isinstance(message, list):
        message = " ".join(message)
    answer, sources = cached_answer_tuple(message)
    log_chat_to_csv(message, answer, sources)
    return ChatResponse(answer=answer, sources=sources)

frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../frontend/out'))
if os.path.isdir(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
    print("Mounted frontend from:", frontend_path)

# load JSONs at startup
filenames = ["unh_catalog.json"]
for name in filenames:
    path = DATA_DIR / name
    if path.exists():
        load_json_file(str(path))
    else:
        print(f"WARNING: {path} not found, skipping.")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8003, reload=True)