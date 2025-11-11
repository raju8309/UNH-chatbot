import re
from functools import lru_cache
from typing import List, Tuple, Dict, Optional

from models.ml_models import get_qa_pipeline
from services.chunk_service import get_chunks_data
from services.retrieval_service import search_chunks
from services.beam_search import generate_with_beam_search
from text_fragments import build_text_fragment_url, choose_snippet, is_synthetic_label
from utils.course_utils import extract_course_fallbacks
from config.settings import get_config
from services.query_enhancement import OpenSourceQueryEnhancer
from services.compression_service import OpenSourceCompressor
from services.reranking_service import OpenSourceReranker
from utils.course_utils import extract_course_fallbacks

# NEW: calendar fallback import
from services.calendar_fallback import maybe_calendar_fallback

UNKNOWN = "I don't have that information."

# lazy-loaded global service instances
_enhancer = None
_compressor = None
_reranker = None

def get_enhancer():
    global _enhancer
    if _enhancer is None:
        _enhancer = OpenSourceQueryEnhancer()
    return _enhancer

def get_compressor():
    global _compressor
    if _compressor is None:
        _compressor = OpenSourceCompressor()
    return _compressor

def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = OpenSourceReranker()
    return _reranker

def _extract_best_credits(chunks):
    for text, meta in chunks:
        match = re.search(r"(\d{1,3})\s*(credits|credit hours?)", text, re.I)
        if match:
            return text, meta, match.group(1)
    return None


def _wrap_sources_with_text_fragments(
    sources_with_passages: List[Tuple[str, Dict]],
    question: str
) -> List[Tuple[str, Dict]]:
    wrapped = []
    for passage, src in sources_with_passages:
        url = src.get("url", "")
        if not url or is_synthetic_label(passage):
            wrapped.append((passage, {**src, "url": url}))
            continue
        
        snippet = choose_snippet(passage, hint=question, max_chars=160)
        wrapped.append((passage, {
            **src,
            "url": build_text_fragment_url(url, text=snippet) if snippet else url
        }))
    return wrapped

def get_prompt(question: str, context: str) -> str:
    return (
        "Using ONLY the provided context, write a concise explanation in exactly 2–3 complete sentences.\n"
        "Mention requirements, deadlines, or procedures if they are present.\n"
        f"If the context is insufficient, output exactly: {UNKNOWN}\n"
        "Do not include assumptions, examples, or general knowledge beyond the context.\n\n"
        f"Context: {context}\n\n"
        f"Question: {question}\n\n"
        "Detailed explanation:"
    )

def _clean_answer(answer: str) -> str:
    """
    Post-process the model answer to remove junk and enforce length limits.
    """
    if not answer or answer.strip() == UNKNOWN:
        return answer
    
    # Remove source markers like [Source 1], [Source 2], etc.
    answer = re.sub(r'\[Source \d+\]', '', answer)
    answer = re.sub(r'\bSource \d+\b', '', answer)
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
    
    # Keep only first 2-3 sentences
    if len(sentences) > 3:
        answer = ' '.join(sentences[:3])
    
    # If still too long (>400 chars), truncate at sentence boundary
    if len(answer) > 400:
        sentences = re.split(r'(?<=[.!?])\s+', answer)
        answer = sentences[0]
        if len(sentences) > 1 and len(answer + ' ' + sentences[1]) <= 400:
            answer = answer + ' ' + sentences[1]
    
    return answer.strip()

# Simple fallbacks without intent detection
def _apply_fallbacks(answer, question, top_chunks):
    def _looks_idk(a: str) -> bool:
        return (a or "").strip() == UNKNOWN
    # degree credits fallback
    qn_lower = (question or "").lower()
    if any(tok in qn_lower for tok in ["credits required", "how many credits", "total credits", "credit requirement"]):
        hit = _extract_best_credits(top_chunks)
        if hit:
            _, _, num = hit
            if _looks_idk(answer) or not re.search(r"\b\d{1,3}\b", answer):
                answer = f"{num}."
    # course fallbacks (simple pattern matching)
    if re.search(r"\b[A-Z]{2,4}\s*\d{3,4}\b", question):
        cf = extract_course_fallbacks(top_chunks)
        need_help = _looks_idk(answer) or \
                   (not re.search(r"credits|prereq|grade", answer, re.I))
        if need_help and any(cf.values()):
            parts = []
            if cf["credits"]:
                parts.append(f"Credits: {cf['credits']}")
            if cf["prereqs"]:
                parts.append(f"Prerequisite(s): {cf['prereqs']}")
            if cf["grademode"]:
                parts.append(f"Grade Mode: {cf['grademode']}")
            if parts:
                answer = ". ".join(parts) + "."
    return answer

def _answer_question(question: str, use_enhancements: bool = True) -> Tuple[str, List[str], List[Dict]]:
    qa_pipeline = get_qa_pipeline()
    cfg = get_config()

    if use_enhancements:
        enhancer = get_enhancer()
        enhanced = enhancer.enhance_query(question, query_type='general')
        question = enhanced["rewritten"]
        print(f"Enhanced query: {question}")

    _, chunk_texts, chunk_sources, _ = get_chunks_data()
    topn_cfg = cfg.get("search", {})
    topn = int(topn_cfg.get("topn_default", 40))
    k_final = int(cfg.get("k", 5))
    retrieval_k = min(topn, 20) if use_enhancements else k_final
    idxs, retrieval_path = search_chunks(
        question,
        topn=topn,
        k=retrieval_k)

    if not idxs:
        # Apply internal fallbacks first
        answer = _apply_fallbacks(UNKNOWN, question, [])

        # Calendar fallback (no sources available in this branch)
        cal_fb = maybe_calendar_fallback(question, answer, [])
        if cal_fb:
            return cal_fb, [], [], None

        return answer, [], [], None
    
    # log if gold chunk was retrieved
    has_gold = any(entry.get("is_gold", False) for entry in retrieval_path)
    if has_gold:
        gold_entries = [e for e in retrieval_path if e.get("is_gold")]
        print(f"Gold chunks in results: {len(gold_entries)}")
        for entry in gold_entries[:2]:
            print(f"  - Rank {entry.get('rank')}: {entry.get('gold_id', 'unknown')}")

    if use_enhancements and len(idxs) > k_final:
        reranker = get_reranker()
        chunks_for_rerank = [(chunk_texts[i], chunk_sources[i]) for i in idxs]
        semantic_scores = [p.get("score", 0.5) for p in retrieval_path if p.get("idx") in idxs]

        reranked_indices = reranker.rerank(
            question,
            chunks_for_rerank,
            semantic_scores,
            use_cross_encoder=True,
            use_tfidf=True,
            top_k=k_final * 2
        )

        final_indices = reranker.diversity_filter(
            chunks_for_rerank,
            reranked_indices,
            diversity_threshold=0.7
        )[:k_final]

        idxs = [idxs[i] for i in final_indices]
        for new_rank, idx in enumerate(idxs, 1):
            for path_entry in retrieval_path:
                if path_entry.get("idx") == idx:
                    path_entry["rank"] = new_rank
                    path_entry["reranked"] = True
    
    top_chunks = [(chunk_texts[i], chunk_sources[i]) for i in idxs]
    if use_enhancements:
        compressor = get_compressor()
        top_chunks = compressor.deduplicate_content(top_chunks)
        top_chunks = compressor.compress_chunks(
            question, top_chunks, max_chunks=k_final, aggressive=False
        )
    top_chunks, context = build_context_string(top_chunks)

    prompt = get_prompt(question, context)
    beam_cfg = cfg.get("beam_search", {})
    use_beam = beam_cfg.get("enabled", True)
    max_tokens = int(cfg.get("performance", {}).get("max_tokens", 200))
    qa_pipeline = get_qa_pipeline()
    
    if use_beam:
        answer = generate_with_beam_search(qa_pipeline, prompt, question, context)
        if answer is None:
            # Beam search failed, use deterministic fallback
            result = qa_pipeline(
                prompt,
                max_new_tokens=max_tokens,
                temperature=0.1,  # Low temperature for consistency
                top_p=0.9,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                do_sample=True  # Minimal sampling for slight variation
            )
            answer = result[0]["generated_text"].strip()
    else:
        # Beam search disabled, use low-temperature generation for consistency
        result = qa_pipeline(
            prompt,
            max_new_tokens=max_tokens,
            temperature=0.1,  # Very low temperature - more deterministic
            top_p=0.9,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            do_sample=True
        )
        answer = result[0]["generated_text"].strip()

    # Clean up the answer (remove source markers, limit length)
    answer = _clean_answer(answer)
    
    # Apply internal fallbacks
    answer = _apply_fallbacks(answer, question, top_chunks)

    # Collect readable titles for calendar-fallback signal
    try:
        source_titles = [src.get("title") or src.get("name") or "" for _, src in top_chunks if isinstance(src, dict)]
    except Exception:
        source_titles = []

    # Calendar fallback last (only when the model didn't provide a concrete deadline/term date)
    cal_fb = maybe_calendar_fallback(question, answer, source_titles)
    if cal_fb:
        # No citations for a pure calendar link response
        return cal_fb, [], retrieval_path, context

    # Build citations
    citation_lines = build_citations(question, top_chunks, retrieval_path)

    return answer, citation_lines, retrieval_path, context

def build_context_string(top_chunks: List[Tuple[str, Dict]]) -> Tuple[List[Tuple[str, Dict]], str]:
    """
    Build context with better structure to help LLM extract key information.
    For Q&A chunks, extract only the Answer portion to reduce verbosity.
    """
    parts = []
    
    # Sort chunks: synthetic Q&A first, then by relevance
    sorted_chunks = []
    qa_chunks = []
    regular_chunks = []
    
    for text, source in top_chunks:
        if "Question:" in text and "Answer:" in text:
            qa_chunks.append((text, source))
        else:
            regular_chunks.append((text, source))
    
    # Q&A format chunks first (they're more direct)
    sorted_chunks = qa_chunks + regular_chunks
    
    for i, (text, source) in enumerate(sorted_chunks, 1):
        title = source.get("title", "Source")
        title = title.split(" - ")[-1] if " - " in title else title
        
        # For Q&A chunks, extract only the answer to reduce context bloat
        display_text = text
        if "Question:" in text and "Answer:" in text:
            # Split on "Answer:" and take everything after it
            answer_parts = text.split("Answer:", 1)
            if len(answer_parts) == 2:
                display_text = answer_parts[1].strip()
        
        parts.append(f"[Source {i}] {title}\n{display_text}")
    
    return sorted_chunks, "\n\n".join(parts)

def build_citations(question, chunks: List[Tuple[str, Dict]], retrieval_path: List[Dict]) -> List[str]:
    enriched_sources = _wrap_sources_with_text_fragments(chunks, question)
    enriched_all = enriched_sources + chunks[3:]

    seen = set()
    citation_lines = []

    for i, (_, src) in enumerate(enriched_all):
        key = (src.get("title"), src.get("url"))
        if key in seen:
            continue
        seen.add(key)

        # check if gold-boosted
        is_gold = False
        if i < len(retrieval_path):
            path_entry = retrieval_path[i]
            is_gold = path_entry.get("is_gold", False)

        # clean title — remove "Gold Q&A:" or any ID suffix
        title = src.get("title", "Source")
        title = re.sub(r"^Gold Q&A:\s*", "", title)
        title = re.sub(r"[:\-]\s*q\d+$", "", title)
        title = title.strip()

        # build the line
        prefix = "- " if is_gold else "- "
        line = f"{prefix}{title}"

        # add link if present
        if src.get("url"):
            line += f" ({src['url']})"

        citation_lines.append(line)

    return citation_lines

@lru_cache(maxsize=128)
def cached_answer_with_path(message: str) -> Tuple[str, List[str], List[Dict], Optional[str]]:
    cfg = get_config()
    use_enhancements = cfg.get("enhancements", {}).get("enabled", True)
    return _answer_question(
        message,
        use_enhancements=use_enhancements
    )