#!/usr/bin/env python3
"""
Manual smoke test for the Query Transformer (LLM rewrite always ON).
- Reads automation_testing/manual_transform_inputs.jsonl
- Prints ORIGINAL -> TRANSFORMED
- Saves/updates results to:
    automation_testing/reports/manual_transform_results.json
    automation_testing/reports/manual_transform_results.csv
Adds metadata:
    kept_tokens_ok
    forbidden_hit
    dup_words_fixed
    fallback_reason
"""

import sys, json, re, csv
from pathlib import Path
from datetime import datetime

# -------- repo import bootstrap --------
THIS = Path(__file__).resolve()
REPO = THIS.parent.parent
BACKEND = REPO / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

# -------- project imports --------
from config import settings
from services.query_transform_service import (
    transform_query,
    apply_domain_rules,
    normalize_text,
    _content_tokens,
    _has_dup_words
)

# Always keep LLM rewrite ON for this script
setattr(settings, "ENABLE_QUERY_REWRITER", True)

INPUT_PATH = REPO / "automation_testing" / "manual_transform_inputs.jsonl"
OUT_JSON   = REPO / "automation_testing" / "reports" / "manual_transform_results.json"
OUT_CSV    = REPO / "automation_testing" / "reports" / "manual_transform_results.csv"

COURSE_CODE_RE = re.compile(r"\b[A-Z]{2,4}\s?\d{3}[A-Z]?\b")

# Fallback forbidden phrases and constraints (pull from settings if present)
FORBIDDEN_PHRASES = getattr(settings, "FORBIDDEN_PHRASES", {"external graduate transfer credits"})
REWRITE_MIN = getattr(settings, "REWRITE_MIN_WORDS", 7)
REWRITE_MAX = getattr(settings, "REWRITE_MAX_WORDS", 18)

STOPWORDS = {"what","is","are","the","a","an","in","for","to","of","and","or","at","on","vs","does","do","can","how","many","long"}

def _now_iso():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def _ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def _read_inputs():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")
    queries = []
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            q = obj.get("query") or obj.get("text") or obj.get("question")
            if not q:
                continue
            q = q.strip()
            if COURSE_CODE_RE.search(q):
                continue
            queries.append(q)
    return queries

def _read_json(path: Path):
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def _write_json(path: Path, data: dict):
    _ensure_parent(path)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

def _score_meta(original: str, transformed: str) -> dict:
    # Determine if this is just rule expansion (fallback) or accepted LLM rewrite
    rule_version = apply_domain_rules(normalize_text(original))
    is_rule_only = (transformed == rule_version)

    orig_tokens = _content_tokens(original)
    rew_tokens  = _content_tokens(transformed)

    kept_tokens_ok = (not orig_tokens) or bool(orig_tokens & rew_tokens)

    low = transformed.lower()
    forbidden_hit = any(p in low for p in FORBIDDEN_PHRASES if "transfer" not in orig_tokens)

    dup_words_fixed = not _has_dup_words(transformed)

    wcount = len(re.findall(r"[a-zA-Z]+", transformed))

    fallback_reason = ""
    if is_rule_only:
        fallback_reason = "rule_fallback"
    else:
        if not kept_tokens_ok:
            fallback_reason = "lost_tokens"
        elif forbidden_hit:
            fallback_reason = "forbidden_phrase"
        elif not dup_words_fixed:
            fallback_reason = "duplicate_words"
        elif not (REWRITE_MIN <= wcount <= REWRITE_MAX):
            fallback_reason = "length_out_of_range"

    return {
        "kept_tokens_ok": kept_tokens_ok,
        "forbidden_hit": forbidden_hit,
        "dup_words_fixed": dup_words_fixed,
        "fallback_reason": fallback_reason
    }

def _write_csv(path: Path, data: dict):
    _ensure_parent(path)
    fieldnames = [
        "original",
        "transformed",
        "timestamp",
        "kept_tokens_ok",
        "forbidden_hit",
        "dup_words_fixed",
        "fallback_reason"
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for k, rec in data.items():
            row = {fn: rec.get(fn, "") for fn in fieldnames}
            w.writerow(row)

def main():
    queries = _read_inputs()

    print("=" * 88)
    print("Query Transformer Smoke Test | LLM rewrite: ON")
    print("=" * 88)

    results = []
    for q in queries:
        tq = transform_query(q)
        meta = _score_meta(q, tq)
        row = {
            "original": q,
            "transformed": tq,
            "timestamp": _now_iso(),
            **meta
        }
        results.append(row)
        print(f"{q:<44} -> {tq}")

    # --- merge to JSON ---
    existing_json = _read_json(OUT_JSON)
    created = updated = 0
    for r in results:
        k = r["original"]
        if k in existing_json:
            existing_json[k].update(r)
            updated += 1
        else:
            existing_json[k] = r
            created += 1
    _write_json(OUT_JSON, existing_json)
    _write_csv(OUT_CSV, existing_json)

    print()
    print(f"Saved {created} new, updated {updated}")
    print(f"JSON: {OUT_JSON}")
    print(f"CSV : {OUT_CSV}")
    print("Done.")

if __name__ == "__main__":
    main()