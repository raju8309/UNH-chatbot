import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple
from models.ml_models import get_qa_pipeline
from services.chunk_service import get_chunks_data
from services.retrieval_service import search_chunks
from text_fragments import build_text_fragment_url, choose_snippet, is_synthetic_label
from utils.course_utils import extract_course_fallbacks
from utils.program_utils import same_program_family
from config.settings import get_config

UNKNOWN = "I don't have that information."

def _wrap_sources_with_text_fragments(
    sources_with_passages: List[Tuple[str, Dict]],
    question: str
) -> List[Dict]:
    wrapped = []
    for passage, src in sources_with_passages:
        url = src.get("url", "")
        if not url or is_synthetic_label(passage):
            wrapped.append({**src, "url": url})
            continue
        
        snippet = choose_snippet(passage, hint=question, max_chars=160)
        wrapped.append({
            **src,
            "url": build_text_fragment_url(url, text=snippet) if snippet else url
        })
    return wrapped

def _extract_best_credits(
    chunks: List[Tuple[str, Dict]]
) -> Optional[Tuple[str, Dict, str]]:
    credit_rx = re.compile(
        r"(?:(?:minimum|at least|a total(?: of)?|total(?: of)?)\s+)?(\d{1,3})\s*(?:credit|credits|cr)\b",
        re.IGNORECASE,
    )
    
    best: Optional[Tuple[str, Dict, str, int]] = None
    
    for text, src in chunks:
        for m in credit_rx.finditer(text or ""):
            num = m.group(1)
            span_text = text[max(0, m.start()-60): m.end()+60]
            
            weight = 1
            if re.search(r"\bminimum\b|\brequired\b|\btotal\b", span_text, re.I):
                weight += 2
            
            try:
                n = int(num)
                if 6 <= n <= 90:
                    weight += 1
            except Exception:
                pass
            
            ans = f"{num}"
            cand = (text, src, ans, weight)
            
            if best is None or cand[3] > best[3]:
                best = cand
    
    if best:
        return (best[0], best[1], best[2])
    return None


def _extract_gre_requirement(
    question: str,
    chunks: List[Tuple[str, Dict]]
) -> Optional[Tuple[str, Dict, str]]:
    qn = question.lower()
    if not any(tok in qn for tok in ["gre", "g r e", "gmat", "g m a t", "test score", "test scores"]):
        return None
    
    for text, src in chunks:
        t = (text or "").lower()
        if "gre" in t or "gmat" in t or "test score" in t or "test scores" in t:
            if re.search(r"\bnot required\b|\bno gre\b|\bwaived\b|\bno (?:gmat|gre) required\b", t):
                return (text, src, "No")
            if re.search(r"\brequired\b|\bmust submit\b|\bofficial scores\b", t):
                return (text, src, "Yes")
    
    return None

def build_context_from_indices(idxs: List[int]) -> Tuple[List[Tuple[str, Dict[str, Any]]], str]:
    _, chunk_texts, chunk_sources, _ = get_chunks_data()
    if not idxs:
        return [], ""
    top_chunks = [(chunk_texts[i], chunk_sources[i]) for i in idxs if i < len(chunk_texts)]
    parts = []
    for text, source in top_chunks:
        title = source.get("title", "Source")
        title = title.split(" - ")[-1] if " - " in title else title
        parts.append(f"{title}: {text}")
    return top_chunks, "\n\n".join(parts)

def get_prompt(question: str, context: str) -> str:
    return (
        "Using ONLY the provided context, write a concise explanation in exactly 2–3 complete sentences.\n"
        "Mention requirements, deadlines, or procedures if they are present.\n"
        f"If the context is insufficient, output exactly: {UNKNOWN}\n"
        "Do not include assumptions, examples, or general knowledge beyond the context.\n\n"
        f"Context: {context}\n\n"
        f"Question: {question}\n\n"
        f"Detailed explanation:"
    )

def _answer_question(
    question: str,
    alias_url: Optional[str] = None,
    intent_key: Optional[str] = None,
    course_norm: Optional[str] = None,
) -> Tuple[str, List[str], List[Dict]]:
    _, chunk_texts, chunk_sources, _ = get_chunks_data()
    qa_pipeline = get_qa_pipeline()
    cfg = get_config()

    topn_cfg = cfg.get("search", {})
    topn_local = int(topn_cfg.get("topn_with_alias", 80)) if alias_url else int(topn_cfg.get("topn_base", 40))
    # honor YAML k
    k_local = int(cfg.get("retrieval_sizes", {}).get("k", cfg.get("k", 5)))
    idxs, retrieval_path = search_chunks(
        question, topn=topn_local, k=k_local, alias_url=alias_url, intent_key=intent_key, course_norm=course_norm
    )
    top_chunks, context = build_context_from_indices(idxs)

    def _url_same_family(u: str, base: Optional[str]) -> bool:
        return bool(base) and same_program_family(u or "", base or "")

    def _has_direct_fact(text: str) -> bool:
        t = (text or "")[:500]
        return bool(re.search(r"\b\d{1,3}\b", t)) or bool(re.search(r"\b(credit|credits|gpa|gre|thesis|option)\b", t, re.I))

    if alias_url and idxs:
        top5 = idxs[:5]
        prefer_idx = None
        for i in top5:
            src = (chunk_sources[i] if i < len(chunk_sources) else {}) or {}
            if _url_same_family(src.get("url",""), alias_url) and _has_direct_fact(chunk_texts[i]):
                prefer_idx = i
                break
        if prefer_idx is not None and prefer_idx != idxs[0]:
            idxs.remove(prefer_idx)
            idxs.insert(0, prefer_idx)
            for r in retrieval_path:
                if r.get("idx") == prefer_idx:
                    r["rank"] = 1
            rank_counter = 2
            for r in retrieval_path:
                if r.get("idx") != prefer_idx:
                    r["rank"] = rank_counter
                    rank_counter += 1
            top_chunks, context = build_context_from_indices(idxs)

    if not idxs:
        return "I couldn't find relevant information in the catalog.", [], retrieval_path, None
    # generate answer
    try:
        result = qa_pipeline(
            get_prompt(question, context),
            max_new_tokens=128,
            repetition_penalty=1.2,
            no_repeat_ngram_size=2,
        )
        answer = result[0]["generated_text"].strip()
    except Exception as exc:
        return f"ERROR running local model: {exc}", [], retrieval_path
    # apply fallbacks
    def _looks_idk(a: str) -> bool:
        return bool(re.search(r"\bi don'?t know\b", (a or "").lower()))

    enriched_sources = _wrap_sources_with_text_fragments(top_chunks, question)

    # degree credits fallback
    if intent_key == "degree_credits":
        hit = _extract_best_credits(top_chunks)
        if hit:
            _, src, num = hit
            if _looks_idk(answer) or not re.search(r"\b\d{1,3}\b", answer):
                answer = f"{num}."
    # GRE fallback
    asked_gre = bool(re.search(r"\b(gre|gmat|test score|test scores)\b", (question or "").lower()))
    gre_hit = _extract_gre_requirement(question, top_chunks)
    if gre_hit:
        _, _, yesno = gre_hit
        if _looks_idk(answer):
            answer = f"{yesno}."
    elif asked_gre and (intent_key == "admissions"):
        answer = ("GRE requirements are program-specific. Many UNH graduate programs do not require GRE, "
                  "but some do. Check the Admission Requirements section on your program page for the current policy.")

    # course fallbacks
    if course_norm and intent_key == "course_info":
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
    enriched_all = enriched_sources + [src for _, src in top_chunks[3:]]
    seen = set()
    citation_lines = []
    for src in enriched_all:
        key = (src.get("title"), src.get("url"))
        if key in seen:
            continue
        seen.add(key)
        line = f"- {src.get('title', 'Source')}"
        if src.get("url"):
            line += f" ({src['url']})"
        citation_lines.append(line)
    return answer, citation_lines, retrieval_path, context

@lru_cache(maxsize=128)
def _cached_answer_core(cache_key: str) -> Tuple[str, List[str], List[Dict]]:
    msg, alias, intent, course = cache_key.split("|||", 3)
    alias = alias or None
    intent = intent or None
    course = course or None
    return _answer_question(msg, alias_url=alias, intent_key=intent, course_norm=course)

def cached_answer_with_path(
    message: str,
    alias_url: Optional[str] = None,
    intent_key: Optional[str] = None,
    course_norm: Optional[str] = None,
) -> Tuple[str, List[str], List[Dict]]:
    cache_key = f"{message}|||{alias_url or ''}|||{intent_key or ''}|||{course_norm or ''}"
    return _cached_answer_core(cache_key)