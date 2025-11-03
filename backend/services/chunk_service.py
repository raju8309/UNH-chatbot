import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from hierarchy import compute_tier
from models.ml_models import get_embed_model
from utils.program_utils import build_program_index
from services.gold_set_service import get_gold_manager, initialize_gold_set  # NEW IMPORTS

# paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "scraper"
CACHE_PATH = DATA_DIR / "chunks_cache.pkl"

# global data structures
chunks_embeddings: Optional[np.ndarray] = None
chunk_texts: List[str] = []
chunk_sources: List[Dict[str, Any]] = []
chunk_meta: List[Dict[str, Any]] = []
CHUNK_NORMS: Optional[np.ndarray] = None

def _compute_meta_from_source(src: Dict[str, Any]) -> Dict[str, Any]:
    return compute_tier(src.get("url", ""), src.get("title", ""))

def save_chunks_cache() -> None:
    global chunks_embeddings, chunk_texts, chunk_sources, chunk_meta

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
                "norms": CHUNK_NORMS
            },
            f,
        )

    print(f"Saved {len(chunk_texts)} chunks to cache")

def load_chunks_cache() -> bool:
    global chunk_texts, chunk_sources, chunks_embeddings, chunk_meta, CHUNK_NORMS
    if not CACHE_PATH.exists():
        return False

    with open(CACHE_PATH, "rb") as f:
        data = pickle.load(f)

    chunk_texts = data.get("texts", [])
    chunk_sources = data.get("sources", [])
    chunks_embeddings = data.get("embeddings")
    CHUNK_NORMS = data.get("norms")
    cached_meta = data.get("meta")

    if cached_meta:
        chunk_meta = cached_meta
    else:
        # recompute metadata if not in cache
        chunk_meta = [_compute_meta_from_source(src) for src in chunk_sources]

    if chunks_embeddings is None or not chunk_texts:
        chunk_meta = []
        chunk_sources = []
        chunk_texts = []
        return False

    if CHUNK_NORMS is None and chunks_embeddings is not None:
        CHUNK_NORMS = np.linalg.norm(chunks_embeddings, axis=1)

    has_gold = any(m.get("tier") == 0 for m in chunk_meta)
    if has_gold:
        print("WARNING: Cache contains gold chunks from previous run")

        # check current config
        from config.settings import get_config
        cfg = get_config()
        gold_enabled = cfg.get("gold_set", {}).get("enabled", True)

        if not gold_enabled:
            print("Gold set is DISABLED, but cache has gold chunks")
            print("Rebuilding index from scratch to exclude gold chunks...")
            return False  # Force rebuild without gold

    print(f"Loaded {len(chunk_texts)} chunks from cache.")
    return True

def load_json_file(path: str) -> None:
    global chunks_embeddings, chunk_texts, chunk_sources, chunk_meta

    embed_model = get_embed_model()
    page_count = 0
    new_texts: List[str] = []
    new_sources: List[Dict[str, Any]] = []
    new_meta: List[Dict[str, Any]] = []

    def add_chunk(text: str, title: str, url: str) -> None:
        word_count = len(text.split())
        has_sentence_punct = any(p in text for p in ".!?")
        min_length = 30
        if word_count >= 3 and (has_sentence_punct or len(text) >= min_length):
            new_texts.append(text)
            src = {"title": title, "url": url}
            new_sources.append(src)
            new_meta.append(_compute_meta_from_source(src))

    def process_section(section: Dict[str, Any], parent_title: str = "", base_url: str = "") -> None:
        nonlocal page_count
        if not isinstance(section, dict):
            return

        page_count += 1

        title = section.get("title", "").strip()
        url = section.get("page_url", base_url).strip()

        full_title = f"{parent_title} - {title}" if parent_title and title else (title or parent_title)

        texts = section.get("text", [])
        if texts:
            for text_item in texts:
                if isinstance(text_item, str):
                    stripped = text_item.strip()
                    if stripped:
                        add_chunk(stripped, full_title, url)

        lists = section.get("lists", [])
        if lists:
            for list_group in lists:
                if isinstance(list_group, list):
                    for item in list_group:
                        if isinstance(item, str):
                            stripped = item.strip()
                            if stripped:
                                add_chunk(stripped, full_title, url)

        subsections = section.get("subsections", [])
        if subsections:
            for subsection in subsections:
                process_section(subsection, full_title, url)

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"ERROR: Could not parse JSON from {path}")
        return

    if isinstance(data, dict):
        pages = data.get("pages", [])
        if isinstance(pages, list):
            for page in pages:
                if isinstance(page, dict):
                    page_title = page.get("page_title", "")
                    page_url = page.get("page_url", "")
                    sections = page.get("sections", [])
                    if isinstance(sections, list):
                        for section in sections:
                            process_section(section, page_title, page_url)

    if new_texts:
        new_embeds = embed_model.encode(new_texts, convert_to_numpy=True)

        if chunks_embeddings is None:
            chunks_embeddings = new_embeds
        else:
            chunks_embeddings = np.vstack([chunks_embeddings, new_embeds])

        chunk_texts.extend(new_texts)
        chunk_sources.extend(new_sources)
        chunk_meta.extend(new_meta)
        global CHUNK_NORMS
        if chunks_embeddings is not None:
            CHUNK_NORMS = np.linalg.norm(chunks_embeddings, axis=1)
        print(f"Loaded {len(new_texts)} chunks from {path}")
    else:
        print(f"WARNING: No text found in {path}")
    print(f"[loader] Pages parsed: {page_count}")
    print(f"[loader] New chunks: {len(new_texts)}  |  Total chunks: {len(chunk_texts)}")

def load_catalog(path: Path) -> None:
    if path.exists():
        load_json_file(str(path))
    else:
        print(f"WARNING: {path} not found, skipping.")

def load_initial_data() -> None:
    global chunk_texts, chunk_sources, chunk_meta, chunks_embeddings, CHUNK_NORMS

    loaded_from_cache = load_chunks_cache()

    if not loaded_from_cache:
        print("Loading data from source files...")
        filenames = ["unh_catalog.json"]
        for name in filenames:
            load_catalog(DATA_DIR / name)
        save_chunks_cache()

    # initialize gold set manager
    embed_model = get_embed_model()
    gold_manager = initialize_gold_set(embed_model)

    # add gold set documents to index
    gold_docs = gold_manager.get_gold_documents()

    if gold_docs:
        print(f"\n=== Integrating {len(gold_docs)} gold Q&A pairs ===")

        gold_texts = []
        gold_sources = []
        gold_meta = []

        for doc in gold_docs:
            gold_texts.append(doc.page_content)
            source = {
                "title": f"Gold Q&A: {doc.metadata.get('gold_id', 'unknown')}",
                "url": doc.metadata.get('url', ''),
            }
            gold_sources.append(source)

            meta = {
                "tier": 0,
                "tier_name": "gold_set",
                "is_program_page": False,
                "level": "graduate",
                "section": doc.metadata.get('gold_id', ''),
                "is_gold": True,
                "original_query": doc.metadata.get('original_query', ''),
                "gold_passages": doc.metadata.get('gold_passages', [])
            }
            gold_meta.append(meta)

        gold_embeddings = embed_model.encode(gold_texts, convert_to_numpy=True)

        if chunks_embeddings is not None:
            chunks_embeddings = np.vstack([gold_embeddings, chunks_embeddings])
        else:
            chunks_embeddings = gold_embeddings

        chunk_texts = gold_texts + chunk_texts
        chunk_sources = gold_sources + chunk_sources
        chunk_meta = gold_meta + chunk_meta

        CHUNK_NORMS = np.linalg.norm(chunks_embeddings, axis=1)

        print(f"Added {len(gold_docs)} gold chunks (Tier 0)")
        print(f"Total chunks now: {len(chunk_texts)}")

    build_program_index(chunk_sources, chunk_meta)

    print("\n=== Gold Set Statistics ===")
    stats = gold_manager.get_statistics()
    print(f"Total gold entries: {stats['total_entries']}")
    print(f"Categories: {stats['categories']}")
    print(f"Has embeddings: {stats['has_embeddings']}")

    print("\n=== Chunk Distribution by Tier ===")
    tier_counts = get_tier_counts()
    for tier in sorted(tier_counts.keys()):
        tier_name = "Gold Set" if tier == 0 else f"Tier {tier}"
        print(f"{tier_name}: {tier_counts[tier]} chunks")

def get_chunks_data() -> tuple:
    return chunks_embeddings, chunk_texts, chunk_sources, chunk_meta

def get_tier_counts() -> Dict[int, int]:
    counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for meta in chunk_meta:
        tier = (meta or {}).get("tier", 4)
        if tier in counts:
            counts[tier] += 1
    return counts

def get_chunk_norms() -> Optional[np.ndarray]:
    global CHUNK_NORMS, chunks_embeddings
    if CHUNK_NORMS is None and chunks_embeddings is not None:
        CHUNK_NORMS = np.linalg.norm(chunks_embeddings, axis=1)
    return CHUNK_NORMS
