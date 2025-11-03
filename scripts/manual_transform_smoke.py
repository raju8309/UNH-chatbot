#!/usr/bin/env python3
"""
Manual smoke test for the Query Transformer (LLM rewrite always ON).
- Reads automation_testing/manual_transform_inputs.jsonl
- Prints ORIGINAL -> TRANSFORMED
- Saves/updates results to both JSON
  automation_testing/reports/manual_transform_results.json
"""

import sys, json, re
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
from services.query_transform_service import transform_query

# Always keep LLM rewrite ON for this script
setattr(settings, "ENABLE_QUERY_REWRITER", True)

INPUT_PATH = REPO / "automation_testing" / "manual_transform_inputs.jsonl"
OUT_JSON   = REPO / "automation_testing" / "reports" / "manual_transform_results.json"

COURSE_CODE_RE = re.compile(r"\b[A-Z]{2,4}\s?\d{3}[A-Z]?\b")

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
            # Skip any accidental course-code queries
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


def main():
    queries = _read_inputs()

    print("=" * 88)
    print("Query Transformer Smoke Test | LLM rewrite: ON")
    print("=" * 88)

    results = []
    for q in queries:
        tq = transform_query(q)
        results.append({"original": q, "transformed": tq, "timestamp": _now_iso()})
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


    print()
    print(f"Saved {created} new, updated {updated}")
    print(f"JSON: {OUT_JSON}")
    print("Done.")

if __name__ == "__main__":
    main()