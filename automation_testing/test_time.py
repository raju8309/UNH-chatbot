
import argparse
import json
import time
import statistics
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL_NAME   = "google/flan-t5-small"

ROOT = Path(__file__).resolve().parents[1]
chunk_texts: List[str] = []
chunk_sources: List[Dict] = []
chunk_ids: List[str] = []
chunk_embeds: Optional[np.ndarray] = None

_embed_model = None
_gen_pipe = None

def _ensure_embed_model():
    from sentence_transformers import SentenceTransformer
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model

def _ensure_gen_pipe():
    from transformers import pipeline
    global _gen_pipe
    if _gen_pipe is None:
        _gen_pipe = pipeline("text2text-generation", model=GEN_MODEL_NAME, device=-1)
    return _gen_pipe

def _doc_id_from_json(data: dict, path: str) -> str:
    from urllib.parse import urlparse
    url = (data or {}).get("url", "")
    if url:
        try:
            p = urlparse(url)
            name = Path(p.path).name or (Path(p.path).parts[-1] if p.path else "")
            if name:
                return name.replace(".html", "").replace(".htm", "")
        except Exception:
            pass
    return Path(path).stem

def _walk_sections(data: dict, path: str):
    """Yield (text, source_dict, chunk_id)."""
    doc_id = _doc_id_from_json(data, path)
    local_idx = 0

    def walk(sections, parent=""):
        nonlocal local_idx
        for sec in sections:
            title = sec.get("title", "") or doc_id
            full_title = f"{parent} > {title}" if parent else title

            for t in sec.get("text", []):
                t = (t or "").strip()
                if not t:
                    continue
                src = {"title": full_title, "url": data.get("url", "")}
                cid = f"{doc_id}#{local_idx}"; local_idx += 1
                yield (t, src, cid)

            for link in sec.get("links", []):
                label = (link or {}).get("label")
                url = (link or {}).get("url")
                if label and url:
                    src = {"title": label, "url": url}
                    cid = f"{doc_id}#{local_idx}"; local_idx += 1
                    yield (f"Courses: {label}", src, cid)

            if "subsections" in sec:
                yield from walk(sec["subsections"], full_title)

    yield from walk(data.get("sections", []))

def _candidate_json_dirs() -> List[Path]:
    return [
        ROOT / "scrape",
        ROOT / "scraper",
        ROOT / "backend" / "scrape",
        ROOT / "backend" / "scraper",
    ]

def _load_all_jsons() -> List[str]:
    paths = []
    for d in _candidate_json_dirs():
        if d.is_dir():
            paths += [str(p) for p in d.glob("*.json")]
    return sorted(set(paths))


def build_index() -> None:
    global chunk_texts, chunk_sources, chunk_ids, chunk_embeds
    json_files = _load_all_jsons()
    if not json_files:
        raise SystemExit("No scraped JSONs found in scrape/ or scraper/. "
                         "Add your catalog files and re-run.")

    texts, sources, ids = [], [], []
    for p in json_files:
        try:
            data = json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception as e:
            print(f"WARNING: skipping {p}: {e}")
            continue
        added = 0
        for t, s, cid in _walk_sections(data, p):
            texts.append(t); sources.append(s); ids.append(cid); added += 1
        print(f"Loaded {added:>4} chunks from {p}")

    if not texts:
        raise SystemExit("No chunks extracted from JSONs. Check file format.")

    model = _ensure_embed_model()
    print(f"Embedding {len(texts)} chunks with {EMBED_MODEL_NAME} …")
    embeds = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    chunk_texts, chunk_sources, chunk_ids, chunk_embeds = texts, sources, ids, embeds


def retrieve(question: str, top_k: int = 5):
    """Cosine similarity over normalized embeddings."""
    assert chunk_embeds is not None and len(chunk_embeds) > 0
    qv = _ensure_embed_model().encode([question], convert_to_numpy=True, normalize_embeddings=True)[0]
    sims = (chunk_embeds @ qv)  
    top = sims.argsort()[-top_k:][::-1]
    return [chunk_ids[i] for i in top], float(sims[top[0]])

def generate(question: str, chunk_ids_sel: List[str]) -> str:
    ctx = " ".join(
        chunk_texts[chunk_ids.index(cid)] 
        for cid in chunk_ids_sel
        if cid in chunk_ids
    )
    prompt = (
        "Answer the question ONLY using the provided context. "
        "If the answer cannot be found, say you don't know.\n\n"
        f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:"
    )
    out = _ensure_gen_pipe()(prompt, max_new_tokens=128)
    return out[0]["generated_text"].strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=20, help="number of measured runs (after warmup)")
    ap.add_argument("--topk", type=int, default=5, help="top-k chunks for context")
    ap.add_argument("--question", type=str,
                    default="Summarize UNH master's degree requirements in one sentence.",
                    help="test query used for latency")
    args = ap.parse_args()

    build_index()


    _ = retrieve(args.question, top_k=args.topk)
    _ = generate(args.question, _[0])

    times = []
    for _i in range(args.runs):
        t0 = time.perf_counter()
        ids, _ = retrieve(args.question, top_k=args.topk)
        _ = generate(args.question, ids)
        times.append(time.perf_counter() - t0)

    times.sort()
    p50 = statistics.median(times)
    p95 = times[int(0.95 * len(times)) - 1]
    p99 = times[int(0.99 * len(times)) - 1]

    print({"runs": args.runs, "p50": p50, "p95": p95, "p99": p99})

if __name__ == "__main__":
    main()
