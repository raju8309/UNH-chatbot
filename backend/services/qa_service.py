import re
from functools import lru_cache
from typing import List, Tuple, Dict, Optional

from models.ml_models import get_qa_pipeline
from services.chunk_service import get_chunks_data
from services.retrieval_service import search_chunks
from services.gold_set_service import get_gold_manager
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

def _extract_gre_requirement(question, chunks):
    gre_terms = ["gre", "gmat", "test score", "test scores"]
    for text, meta in chunks:
        if any(term in text.lower() for term in gre_terms):
            yesno = "Yes" if re.search(r"required|must submit", text, re.I) else "No"
            return text, meta, yesno
    return None

def _looks_idk(answer: str) -> bool:
    return bool(re.search(r"\bi don'?t know\b", (answer or "").lower()))

def apply_fallbacks(answer, question, chunks, intent_key, course_norm):
    # credits fallback
    if intent_key == "degree_credits":
        hit = _extract_best_credits(chunks)
        if hit:
            _, _, num = hit
            if _looks_idk(answer) or not re.search(r"\b\d{1,3}\b", answer):
                answer = f"{num}."

    # GRE fallback
    if intent_key == "admissions":
        asked_gre = bool(re.search(r"\b(gre|gmat|test score|test scores)\b", question.lower()))
        gre_hit = _extract_gre_requirement(question, chunks)
        if gre_hit:
            _, _, yesno = gre_hit
            if _looks_idk(answer):
                answer = f"{yesno}."
        elif asked_gre:
            answer = (
                "GRE requirements are program-specific. Many UNH graduate programs "
                "do not require GRE, but some do. Check the Admission Requirements "
                "section on your program page for the current policy."
            )

    # course fallback
    if course_norm and intent_key == "course_info":
        cf = extract_course_fallbacks(chunks)
        need_help = _looks_idk(answer) or not re.search(r"credits|prereq|grade", answer, re.I)
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

def build_context_string(top_chunks: List[Tuple[str, Dict]]) -> Tuple[List[Tuple[str, Dict]], str]:
    parts = []
    for text, source in top_chunks:
        title = source.get("title", "Source")
        title = title.split(" - ")[-1] if " - " in title else title
        parts.append(f"{title}: {text}")
    return top_chunks, "\n\n".join(parts)

def build_citations(chunks: List[Tuple[str, Dict]], retrieval_path: List[Dict]) -> List[str]:
    seen = set()
    citation_lines = []

    for i, (text, src) in enumerate(chunks):
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

def _answer_question_unified(
    question: str,
    alias_url: Optional[str] = None,
    intent_key: Optional[str] = None,
    course_norm: Optional[str] = None,
    use_enhancements: bool = True
) -> Tuple[str, List[str], List[Dict], Optional[str]]:
    cfg = get_config()
    qa_pipeline = get_qa_pipeline()

    search_query = question
    if use_enhancements:
        enhancer = get_enhancer()
        enhanced = enhancer.enhance_query(question, query_type=intent_key or 'general')
        search_query = enhanced["rewritten"]
        print(f"Enhanced query: {search_query}")

    _, chunk_texts, chunk_sources, _ = get_chunks_data()
    topn_cfg = cfg.get("search", {})
    topn = int(topn_cfg.get("topn_with_alias", 80)) if alias_url else int(topn_cfg.get("topn_base", 40))
    k_final = int(cfg.get("k", 5))
    retrieval_k = min(topn, 20) if use_enhancements else k_final

    idxs, retrieval_path = search_chunks(
        search_query,
        topn=topn,
        k=retrieval_k,
        alias_url=alias_url,
        intent_key=intent_key,
        course_norm=course_norm
    )

    if not idxs:
        # Apply internal fallbacks first
        answer = apply_fallbacks(UNKNOWN, question, [], intent_key, course_norm)

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
    try:
        result = qa_pipeline(
            prompt,
            max_new_tokens=128,
            repetition_penalty=1.2,
            no_repeat_ngram_size=2,
        )
        answer = result[0]["generated_text"].strip()
    except Exception as exc:
        answer = f"ERROR running model: {exc}"

    # Internal fallbacks first
    answer = apply_fallbacks(answer, question, top_chunks, intent_key, course_norm)

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

    citation_lines = build_citations(top_chunks, retrieval_path)

    return answer, citation_lines, retrieval_path, context

@lru_cache(maxsize=128)
def cached_answer_with_path(
    message: str,
    alias_url: Optional[str] = None,
    intent_key: Optional[str] = None,
    course_norm: Optional[str] = None,
) -> Tuple[str, List[str], List[Dict], Optional[str]]:
    cfg = get_config()
    use_enhancements = cfg.get("enhancements", {}).get("enabled", True)
    return _answer_question_unified(
        message,
        alias_url=alias_url,
        intent_key=intent_key,
        course_norm=course_norm,
        use_enhancements=use_enhancements
    )