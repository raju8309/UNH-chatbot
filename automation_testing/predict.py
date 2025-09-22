#!/usr/bin/env python3
"""
Offline predictor: no backend required.

- Loads scraped JSONs from ../scrape or ../scraper
- Embeds all chunks with all-MiniLM-L6-v2
- Retrieves top-5 chunks per question (with a small grading heuristic)
- Generates answers with Flan-T5-Small (or T5-small fallback) locally
- Writes preds.jsonl compatible with evaluator.py
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline,
)

# ----------------- Paths -----------------
ROOT = Path(__file__).resolve().parents[1]
GOLD = Path(__file__).with_name("gold.jsonl")
PREDS = Path(__file__).with_name("preds.jsonl")

# ----------------- Config -----------------
# You can override model names with env vars if needed
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GEN_MODEL_NAME = os.getenv("GEN_MODEL", "google/flan-t5-small")
MODEL_CACHE_DIR = ROOT / ".model_cache"


chunk_texts: List[str] = []
chunk_sources: List[Dict] = []
chunk_ids: List[str] = []
chunk_embeds: Optional[np.ndarray] = None


_EMBEDDER: Optional[SentenceTransformer] = None
_GENERATOR = None
_OFFLINE = False

def _doc_id_from_json(data: dict, path: str) -> str:
    """Prefer URL last segment, else file stem."""
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
    """Yield (text, source_dict, id_str) from the scraped JSON structure."""
    doc_id = _doc_id_from_json(data, path)
    local_idx = 0

    def walk(sections, parent=""):
        nonlocal local_idx
        for sec in sections:
            title = sec.get("title", "") or doc_id
            full_title = f"{parent} > {title}" if parent else title

            # paragraphs
            for t in sec.get("text", []):
                t = (t or "").strip()
                if not t:
                    continue
                src = {"title": full_title, "url": data.get("url", "")}
                cid = f"{doc_id}#{local_idx}"; local_idx += 1
                yield (t, src, cid)

            # links
            for link in sec.get("links", []):
                label = (link or {}).get("label")
                url = (link or {}).get("url")
                if label and url:
                    src = {"title": label, "url": url}
                    cid = f"{doc_id}#{local_idx}"; local_idx += 1
                    yield (f"Courses: {label}", src, cid)

            # subsections
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


# ----------------- Model loaders -----------------
def _load_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        print(f"[info] loading embedder: {EMBED_MODEL_NAME} (offline={_OFFLINE})")
        _EMBEDDER = SentenceTransformer(
            EMBED_MODEL_NAME,
            cache_folder=str(MODEL_CACHE_DIR),
            local_files_only=_OFFLINE,
        )
    return _EMBEDDER


def _load_generator():
    """Load text2text generator with proper offline handling; no invalid kwargs passed to generate()."""
    global _GENERATOR
    if _GENERATOR is not None:
        return _GENERATOR

    def build(gen_name: str):
        print(f"[info] loading generator: {gen_name} (offline={_OFFLINE})")
        tok = AutoTokenizer.from_pretrained(
            gen_name,
            cache_dir=str(MODEL_CACHE_DIR),
            local_files_only=_OFFLINE,
        )
        mdl = AutoModelForSeq2SeqLM.from_pretrained(
            gen_name,
            cache_dir=str(MODEL_CACHE_DIR),
            local_files_only=_OFFLINE,
            low_cpu_mem_usage=True,
        )
        gen = pipeline(
            "text2text-generation",
            model=mdl,
            tokenizer=tok,
            device=-1,  # CPU
        )
        # keep tokenizer for truncation
        gen._tok = tok
        return gen

    try:
        _GENERATOR = build(GEN_MODEL_NAME)
    except Exception as e:
        print(f"[warn] could not load {GEN_MODEL_NAME}: {e}")
        fallback = "t5-small"
        print(f"[info] trying fallback generator: {fallback}")
        _GENERATOR = build(fallback)

    return _GENERATOR


def build_index():
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

    print(f"Embedding {len(texts)} chunks with {EMBED_MODEL_NAME} …")
    embedder = _load_embedder()
    embeds = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    chunk_texts, chunk_sources, chunk_ids, chunk_embeds = texts, sources, ids, embeds


def retrieve(question: str, top_k: int = 5):
    """Cosine sim over normalized embeddings + tiny grading heuristic."""
    assert chunk_embeds is not None and len(chunk_embeds) > 0
    embedder = _load_embedder()
    qv = embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True)[0]
    sims = (chunk_embeds @ qv)

    pool_k = max(top_k, 10)
    pool = sims.argsort()[-pool_k:][::-1]

    ql = question.lower()
    wants_grading = any(k in ql for k in ["grade", "grading", "letter grade", "b-"])
    boosted = []
    for i in pool:
        s = float(sims[i])
        base = (chunk_ids[i] if i < len(chunk_ids) else "").split("#", 1)[0]
        if wants_grading and base == "grading":
            s *= 1.2
        boosted.append((i, s))
    boosted.sort(key=lambda t: t[1], reverse=True)
    keep = [i for i, _ in boosted[:top_k]]

    results = []
    for i in keep:
        results.append({
            "id": chunk_ids[i],
            "text": chunk_texts[i],
            "source": chunk_sources[i],
            "score": float(sims[i]),
        })
    return results


def _truncate_to_model_limit(gen, text: str, reserve_new_tokens: int = 128) -> str:
    """
    Ensure input <= model max length - reserve_new_tokens.
    Keep the end of the context (often the most relevant after concatenation).
    """
    tok = getattr(gen, "_tok", None)
    if tok is None:
        return text

    max_len = getattr(tok, "model_max_length", 512) or 512
    hard_cap = max_len - max(8, reserve_new_tokens)
    ids = tok.encode(text, add_special_tokens=True)
    if len(ids) <= hard_cap:
        return text
    trimmed_ids = ids[-hard_cap:]
    return tok.decode(trimmed_ids, skip_special_tokens=True)


def generate_answer(question: str, top_chunks: List[Dict]) -> Dict:
    gen = _load_generator()
    ctx = " ".join(c["text"] for c in top_chunks)
    prompt = (
        "Answer the question ONLY using the provided context. "
        "If the answer cannot be found, say you don't know.\n\n"
        f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:"
    )

    prompt = _truncate_to_model_limit(gen, prompt, reserve_new_tokens=128)
    out = gen(prompt, max_new_tokens=128)
    answer = out[0]["generated_text"].strip()

    # dedupe sources
    seen, srcs = set(), []
    for c in top_chunks:
        s = c["source"]
        key = (s["title"], s.get("url", ""))
        if key not in seen:
            seen.add(key)
            srcs.append({"title": s["title"], "url": s.get("url", "")})
    ids = [c["id"] for c in top_chunks]
    return {"answer": answer, "sources": srcs, "retrieved_ids": ids}


def read_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="Use only local cached models (no downloads)")
    return ap.parse_args()


def main():
    global _OFFLINE
    args = parse_args()
    _OFFLINE = bool(args.offline)

    if not GOLD.exists():
        raise SystemExit(f"Missing gold file: {GOLD}")

    build_index()

    outs = []
    for row in read_jsonl(GOLD):
        qid = row["id"]
        query = row["query"]
        top = retrieve(query, top_k=5)
        pred = generate_answer(query, top)
        outs.append({
            "id": qid,
            "model_answer": pred["answer"],
            "retrieved_ids": pred["retrieved_ids"],
        })
        print(f"[OK] {qid}  ->  {pred['retrieved_ids'][:3]}")

    PREDS.write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in outs),
        encoding="utf-8",
    )
    print(f"Saved predictions to {PREDS}")


if __name__ == "__main__":
    main()
