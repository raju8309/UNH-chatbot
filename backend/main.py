import csv
import json
import os
import pickle
import re
import threading
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import uvicorn

from dashboard import router as dashboard_router
from hierarchy import compute_tier
from text_fragments import build_text_fragment_url, choose_snippet, is_synthetic_label

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "scraper"
CACHE_PATH = DATA_DIR / "chunks_cache.pkl"
CHAT_LOG_PATH = "chat_logs.csv"

UNKNOWN = "I don't have that information."

CFG: Dict[str, Any] = {}
POLICY_TERMS: Tuple[str, ...] = ()

chunks_embeddings = None
chunk_texts: List[str] = []
chunk_sources: List[Dict[str, Any]] = []
chunk_meta: List[Dict[str, Any]] = []

_LOG_LOCK = threading.Lock()

embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load trained model, fallback on default
trained = Path(__file__).parent / "models" / "flan-t5-small-finetuned"
if trained.exists() and (trained / "config.json").exists():
    model = str(trained)
else:
    model = "google/flan-t5-small"
qa_pipeline = pipeline(
    "text2text-generation",
    model=model,
    device=-1,
)

app = FastAPI()

def load_retrieval_cfg() -> None:
    global CFG, POLICY_TERMS
    cfg_path = Path(__file__).resolve().parent / "config" / "retrieval.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            CFG = yaml.safe_load(f) or {}
    else:
        CFG = {}
    CFG.setdefault("policy_terms", [])
    CFG.setdefault("tier_boosts", {1: 1.35, 2: 1.10, 3: 1.0, 4: 1.0})
    CFG.setdefault(
        "intent",
        {
            "course_keywords": [],
            "degree_keywords": [],
            "course_code_regex": r"\b[A-Z]{3,5}\s?\d{3}\b",
        },
    )
    CFG.setdefault("nudges", {"policy_acadreg_url": 1.15})
    CFG.setdefault("guarantees", {"ensure_tier1_on_policy": True})
    CFG.setdefault(
        "tier4_gate",
        {"use_embedding": True, "min_title_sim": 0.42, "min_alt_sim": 0.38},
    )
    POLICY_TERMS = tuple(CFG.get("policy_terms", []))

def _compute_meta_from_source(src: Dict[str, Any]) -> Dict[str, Any]:
    return compute_tier(src.get("url", ""), src.get("title", ""))

def save_chunks_cache() -> None:
    if not chunk_texts or chunks_embeddings is None:
        return
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(
            {
                "texts": chunk_texts,
                "sources": chunk_sources,
                "embeddings": chunks_embeddings,
                "meta": chunk_meta,
            },
            f,
        )
    print(f"saved {len(chunk_texts)} chunks to cache")

def load_chunks_cache() -> bool:
    global chunk_texts, chunk_sources, chunks_embeddings, chunk_meta
    if not CACHE_PATH.exists():
        return False
    with open(CACHE_PATH, "rb") as f:
        data = pickle.load(f)
    chunk_texts = data.get("texts", [])
    chunk_sources = data.get("sources", [])
    chunks_embeddings = data.get("embeddings")
    cached_meta = data.get("meta")
    if cached_meta:
        chunk_meta = cached_meta
    else:
        chunk_meta = [_compute_meta_from_source(src) for src in chunk_sources]
    if chunks_embeddings is None or not chunk_texts:
        chunk_meta = []
        chunk_sources = []
        chunk_texts = []
        return False
    print(f"Loaded {len(chunk_texts)} chunks from cache.")
    return True

def load_json_file(path: str) -> None:
    global chunks_embeddings, chunk_texts, chunk_sources, chunk_meta

    page_count = 0

    def _process_section(section: Dict[str, Any], parent_title: str = "", base_url: str = "") -> None:
        nonlocal page_count  # Access the outer variable
        """Recursively process sections and subsections"""
        if not isinstance(section, dict):
            return
            
        page_count += 1
        
        title = section.get("title", "").strip()
        url = section.get("page_url", base_url).strip()
        
        # Build hierarchical title
        full_title = f"{parent_title} - {title}" if parent_title and title else (title or parent_title)
        
        # Process text content - add everything no matter length
        texts = section.get("text", [])
        if texts:
            for text_item in texts:
                if isinstance(text_item, str) and text_item.strip():
                    add_chunk(text_item.strip(), full_title, url)
        
        # Process lists - only keep content more than 5 words to filter out navigational lists
        lists = section.get("lists", [])
        if lists:
            for list_group in lists:
                if isinstance(list_group, list):
                    for item in list_group:
                        if isinstance(item, str) and item.strip():
                            if (len(item) > 5):
                                add_chunk(item.strip(), full_title, url)
        
        # Process subsections recursively
        subsections = section.get("subsections", [])
        if subsections:
            for subsection in subsections:
                _process_section(subsection, full_title, url)

    def add_chunk(text: str, title: str, url: str) -> None:
        new_texts.append(text)
        src = {"title": title, "url": url}
        new_sources.append(src)
        new_meta.append(_compute_meta_from_source(src))

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"ERROR: Could not parse JSON from {path}")
        return

    new_texts: List[str] = []
    new_sources: List[Dict[str, Any]] = []
    new_meta: List[Dict[str, Any]] = []

    # Process the root structure
    if isinstance(data, dict):
        root_sections = data.get("sections", [])
        if isinstance(root_sections, list):
            for section in root_sections:
                _process_section(section)
    
    if new_texts:
        new_embeds = embed_model.encode(new_texts, convert_to_numpy=True)
        if chunks_embeddings is None:
            chunks_embeddings = new_embeds
        else:
            chunks_embeddings = np.vstack([chunks_embeddings, new_embeds])
        chunk_texts.extend(new_texts)
        chunk_sources.extend(new_sources)
        chunk_meta.extend(new_meta)
        print(f"Loaded {len(new_texts)} chunks from {path}")
    else:
        print(f"WARNING: no text found in {path}")
    print(f"[loader] Pages parsed: {page_count}")
    print(f"[loader] New chunks: {len(new_texts)}  |  Total chunks: {len(chunk_texts)}")

def load_catalog(path: Path) -> None:
    if path.exists():
        load_json_file(str(path))
    else:
        print(f"WARNING: {path} not found, skipping.")

def _program_intent(query: str) -> bool:
    q = (query or "")
    ql = q.lower()
    intent = CFG.get("intent", {})
    course_kw = intent.get("course_keywords", [])
    degree_kw = intent.get("degree_keywords", [])
    code_rx = intent.get("course_code_regex", r"\b[A-Z]{3,5}\s?\d{3}\b")
    course_code = re.search(code_rx, q)
    return any(k in ql for k in (course_kw + degree_kw)) or bool(course_code)

def _tier_boost(tier: int) -> float:
    return float(CFG.get("tier_boosts", {}).get(tier, 1.0))

def _is_acad_reg_url(url: str) -> bool:
    return isinstance(url, str) and "/graduate/academic-regulations-degree-requirements/" in url

def _title_for_sim(src: Dict[str, Any]) -> str:
    title = (src.get("title") or "").strip()
    url = (src.get("url") or "")
    path = urlparse(url).path if url else ""
    segs = [s for s in path.split("/") if s]
    tail = " ".join(segs[-2:]) if segs else ""
    return (title + " " + tail).strip()

def _tier4_is_relevant_embed(query: str, idx: int) -> bool:
    gate = CFG.get("tier4_gate", {})
    if not gate.get("use_embedding", True):
        return True
    if idx >= len(chunk_sources):
        return False
    cand = _title_for_sim(chunk_sources[idx])
    if not cand:
        return False
    q_vec, c_vec = embed_model.encode([query, cand], convert_to_numpy=True)
    sim = float(np.dot(q_vec, c_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(c_vec) + 1e-8))
    thresh = float(gate.get("min_title_sim", 0.42))
    return sim >= thresh

def search_chunks(query: str, topn: int = 40, k: int = 5):
    if chunks_embeddings is None or not chunk_texts:
        return [], []
    q_vec = embed_model.encode([query], convert_to_numpy=True)[0]
    
    # Compute cosine similarities
    chunk_norms = np.linalg.norm(chunks_embeddings, axis=1)
    query_norm = np.linalg.norm(q_vec)
    
    # Avoid division by zero
    valid_chunks = chunk_norms > 1e-8
    sims = np.zeros(len(chunks_embeddings))
    
    if query_norm > 1e-8:
        sims[valid_chunks] = (chunks_embeddings[valid_chunks] @ q_vec) / (chunk_norms[valid_chunks] * query_norm)
    
    # Get top candidates
    cand_idxs = np.argsort(-sims)[:topn * 2].tolist()
    
    # Enhanced filtering based on query analysis
    q_lower = (query or "").lower()
    allow_program = _program_intent(query)
    looks_policy = any(term in q_lower for term in POLICY_TERMS)
    
    # Extract key terms from query for better matching
    query_terms = set(re.findall(r'\b\w+\b', q_lower))
    
    filtered: List[int] = []
    for i in cand_idxs:
        if i >= len(chunk_texts):
            continue
            
        chunk_text_lower = chunk_texts[i].lower()
        meta_i = chunk_meta[i] if i < len(chunk_meta) else {}
        tier = meta_i.get("tier", 2)
        
        # More lenient relevance check
        term_matches = len(query_terms.intersection(set(re.findall(r'\b\w+\b', chunk_text_lower))))
        if term_matches == 0 and sims[i] < 0.1:  # Lower threshold
            continue
            
        # Apply tier filtering
        if tier in (3, 4) and not allow_program:
            continue
        if tier == 4 and allow_program and not _tier4_is_relevant_embed(query, i):
            continue
        filtered.append(i)

    if not filtered:
        allowed_tiers = {1, 2} if not allow_program else {1, 2, 3, 4}
        filtered = [
            i for i in range(len(chunk_meta))
            if ((chunk_meta[i] or {}).get("tier") in allowed_tiers)
        ] or list(range(len(chunk_meta)))

    policy_nudge = float(CFG.get("nudges", {}).get("policy_acadreg_url", 1.15))
    rescored = []
    for i in filtered:
        meta_i = chunk_meta[i] if i < len(chunk_meta) else {}
        src_i = chunk_sources[i] if i < len(chunk_sources) else {}
        tier = meta_i.get("tier", 2)
        base = float(sims[i]) * _tier_boost(tier)
        nudge = policy_nudge if looks_policy and _is_acad_reg_url(src_i.get("url", "")) else 1.0
        rescored.append((i, base * nudge))
    rescored.sort(key=lambda x: x[1], reverse=True)
    ordered = [i for i, _ in rescored]

    final: List[int] = []
    if looks_policy and bool(CFG.get("guarantees", {}).get("ensure_tier1_on_policy", True)):
        has_tier1 = any((chunk_meta[i] or {}).get("tier") == 1 for i in ordered[:k])
        if not has_tier1:
            best_t1_idx = -1
            best_t1_score = -1.0
            for i in range(len(chunk_meta)):
                meta_i = chunk_meta[i] or {}
                if meta_i.get("tier") == 1:
                    sc = float(sims[i]) * _tier_boost(1) * policy_nudge
                    if sc > best_t1_score:
                        best_t1_score = sc
                        best_t1_idx = i
            if best_t1_idx != -1:
                final.append(best_t1_idx)

    for i in ordered:
        if len(final) >= k:
            break
        if i not in final:
            final.append(i)

    final = final[:k]
    retrieval_path = []
    for rank, i in enumerate(final, start=1):
        src = chunk_sources[i] if i < len(chunk_sources) else {}
        meta = chunk_meta[i] if i < len(chunk_meta) else {}
        retrieval_path.append(
            {
                "rank": rank,
                "idx": i,
                "score": round(float(sims[i]), 6),
                "title": src.get("title"),
                "url": src.get("url"),
                "tier": meta.get("tier"),
                "tier_name": meta.get("tier_name"),
                "text": chunk_texts[i] if i < len(chunk_texts) else ""
            }
        )
    return final, retrieval_path

def _wrap_sources_with_text_fragments(sources_with_passages, question: str):
    wrapped = []
    for passage, src in sources_with_passages:
        url = src.get("url", "")
        if not url or is_synthetic_label(passage):
            wrapped.append({**src, "url": url})
            continue
        snippet = choose_snippet(passage, hint=question, max_chars=160)
        wrapped.append({**src, "url": build_text_fragment_url(url, text=snippet) if snippet else url})
    return wrapped

def get_context(question):
    idxs, retrieval_path = search_chunks(question, topn=40, k=5)
    if not idxs:
        return UNKNOWN, [], retrieval_path
    top_chunks = [(chunk_texts[i], chunk_sources[i]) for i in idxs]
    # Create context with source attribution and new lines
    context_parts = []
    for i, (text, source) in enumerate(top_chunks):
        title = source.get('title', 'Source')
        # Don't use hierarchial title for context, confuses the ai
        title = title.split(' - ')[-1] if ' - ' in title else title
        context_parts.append(f"{title}: {text}")
    return top_chunks, retrieval_path, "\n\n".join(context_parts)

def get_prompt(question, context):
    return (
        "Using ONLY the provided context, write a concise explanation in exactly 2–3 complete sentences.\n"
        "Mention requirements, deadlines, or procedures if they are present.\n"
        f"If the context is insufficient, output exactly: {UNKNOWN}\n"
        "Do not include assumptions, examples, or general knowledge beyond the context.\n\n"
        f"Context: {context}\n\n"
        f"Question: {question}\n\n"
        f"Detailed explanation:"
    )

def _answer_question(question: str):
    top_chunks, retrieval_path, context = get_context(question)
    try:
        long_result = qa_pipeline(
            get_prompt(question, context),
            max_new_tokens=128,
            repetition_penalty=1.2,
            no_repeat_ngram_size=2)
        answer = long_result[0]["generated_text"].strip()
    except Exception as exc:
        return f"ERROR running local model: {exc}", [], retrieval_path

    enriched_sources = _wrap_sources_with_text_fragments(top_chunks, question)
    seen = set()
    citation_lines = []
    for src in enriched_sources:
        key = (src.get("title"), src.get("url"))
        if key in seen:
            continue
        seen.add(key)
        line = f"- {src.get('title', 'Source')}"
        if src.get("url"):
            line += f" ({src['url']})"
        citation_lines.append(line)
    return answer, citation_lines, retrieval_path

@lru_cache(maxsize=128)
def cached_answer_tuple(question_str: str):
    return _answer_question(question_str)

def cached_answer_with_path(message: str):
    return cached_answer_tuple(message)

# small test helper: answer + retrieved_ids
def base_doc_id(url: str) -> str:
    if not url:
        return "catalog"
    p = urlparse(url)
    name = (Path(p.path).name or "").rstrip("/")
    if not name and p.path:
        name = Path(p.path).parts[-1]
    slug = (name or "catalog").replace(".html", "").replace(".htm", "") or "catalog"
    return slug

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[str]] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    retrieval_path: List[Dict[str, Any]]

@app.get("/debug/tier-counts")
def tier_counts():
    counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for meta in chunk_meta:
        tier = (meta or {}).get("tier")
        if tier in counts:
            counts[tier] += 1
    return {
        "tier1": counts[1],
        "tier2": counts[2],
        "tier3": counts[3],
        "tier4": counts[4],
        "total": len(chunk_meta),
    }

@app.post("/reset")
async def reset_chat():
    return {"status": "cleared"}

def log_chat_to_csv(question: str, answer: str, sources: List[str]) -> None:
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
    answer, sources, retrieval_path = cached_answer_with_path(message)
    log_chat_to_csv(message, answer, sources)
    return ChatResponse(answer=answer, sources=sources, retrieval_path=retrieval_path)

def ensure_chat_log_file() -> None:
    if not os.path.isfile(CHAT_LOG_PATH):
        with open(CHAT_LOG_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["timestamp", "question", "answer", "sources_json"])

def configure_app(app_instance: FastAPI) -> None:
    app.include_router(dashboard_router)
    PUBLIC_URL = os.getenv("PUBLIC_URL", "http://localhost:8003/")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[PUBLIC_URL],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    frontend_path = BASE_DIR / "frontend" / "out"
    if frontend_path.is_dir():
        app_instance.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")
        print("Mounted frontend from:", frontend_path)

def load_initial_data() -> None:
    loaded_from_cache = load_chunks_cache()
    if not loaded_from_cache:
        filenames = ["unh_catalog.json"]
        for name in filenames:
            load_catalog(DATA_DIR / name)
        save_chunks_cache()

if __name__ == "__main__":
    load_retrieval_cfg()
    ensure_chat_log_file()
    load_initial_data()
    configure_app(app)
    uvicorn.run(app, host="0.0.0.0", port=8003, reload=False)
