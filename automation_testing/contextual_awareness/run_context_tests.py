#!/usr/bin/env python3
"""
Context-only runner
- Uses backend pipeline to answer Qs in contextual_awareness/context_gold.jsonl
- Writes timestamped folder under contextual_awareness/reports/
- Saves:
    - context_gold.jsonl (copy of input gold)
    - preds.jsonl        (model answers + retrieved ids)
"""

from __future__ import annotations
import sys, json, shutil, subprocess
from pathlib import Path
from datetime import datetime
import argparse

#  Paths 
ROOT = Path(__file__).resolve().parents[2]  
AUTO_DIR = ROOT / "automation_testing"
CTX_DIR  = AUTO_DIR / "contextual_awareness"
GOLD     = CTX_DIR / "context_gold.jsonl"
EVAL     = AUTO_DIR / "evaluator.py" 
# Make backend importable
sys.path.insert(0, str(ROOT / "backend"))

# Import pipeline pieces (align with main run_tests.py) 
from config.settings import load_retrieval_config
from models.ml_models import initialize_models
from services.chunk_service import load_initial_data
from services.query_pipeline import process_question_for_retrieval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true",
                    help="Also run evaluator.py to create report.json (optional).")
    args = ap.parse_args()

    if not GOLD.exists():
        raise SystemExit(f"Missing contextual gold file: {GOLD}")

    # Create timestamped report directory inside contextual_awareness/reports/
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    reports_dir = CTX_DIR / "reports"
    out_dir = reports_dir / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy gold to report folder (named context_gold.jsonl)
    gold_copy = out_dir / "context_gold.jsonl"
    shutil.copy2(GOLD, gold_copy)
    print(f"Copied gold -> {gold_copy}")

    # Initialize the full retrieval + QA pipeline
    print("🔧 Initializing pipeline…")
    load_retrieval_config()
    initialize_models()
    load_initial_data()

    # Generate predictions (mirror run_tests.py output shape)
    preds_path = out_dir / "preds.jsonl"
    count = 0
    with GOLD.open("r", encoding="utf-8") as fin, preds_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            rec = json.loads(line)
            qid = rec.get("id")
            query = rec.get("query", "")

            result = process_question_for_retrieval(query)
            ans = result.get("answer", "")
            retrieved_ids = result.get("retrieval_path", [])

            fout.write(json.dumps({
                "id": qid,
                "model_answer": ans,
                "retrieved_ids": retrieved_ids
            }, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} predictions -> {preds_path}")

    # Optional scoring (off by default)
    if args.score:
        if not EVAL.exists():
            print("evaluator.py not found; skipping scoring.")
        else:
            print("Running evaluator.py…")
            subprocess.check_call([sys.executable, str(EVAL), "--output-dir", str(out_dir)])
            print(f"Scoring output in: {out_dir}")

    print("\nDone. Context report folder:")
    print(f"  {out_dir}")
    print(f"   ├─ context_gold.jsonl")
    print(f"   └─ preds.jsonl")
    if args.score:
        print(f"   └─ report.json")


if __name__ == "__main__":
    main()