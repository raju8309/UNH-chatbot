
"""
Context-only runner
- Uses backend pipeline to answer Qs in contextual_awareness/context_gold.jsonl
- Writes timestamped folder under contextual_awareness/reports/
- Saves:
    - context_gold.jsonl (copy of input gold)
    - preds.jsonl        (model answers + retrieved ids)
- Optional: --score will run automation_testing/evaluator.py into the same folder.
"""

from __future__ import annotations
import sys, json, shutil, subprocess
from pathlib import Path
from datetime import datetime
import argparse

# ----- Paths -----
ROOT = Path(__file__).resolve().parents[2]  # repo root
AUTO_DIR = ROOT / "automation_testing"
CTX_DIR  = AUTO_DIR / "contextual_awareness"
GOLD     = CTX_DIR / "context_gold.jsonl"
EVAL     = AUTO_DIR / "evaluator.py"  # used only if --score is passed

# Make backend importable
sys.path.insert(0, str(ROOT / "backend"))

# ----- Import your pipeline pieces -----
from config.settings import load_retrieval_config
from models.ml_models import initialize_models
from services.chunk_service import load_initial_data
from services.qa_service import _answer_question


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true",
                    help="Also run evaluator.py to create report.json (optional).")
    args = ap.parse_args()

    if not GOLD.exists():
        raise SystemExit(f"❌ Missing contextual gold file: {GOLD}")

    # Create timestamped report directory inside contextual_awareness/reports/
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    reports_dir = CTX_DIR / "reports"
    out_dir = reports_dir / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy gold to report folder (named context_gold.jsonl)
    gold_copy = out_dir / "context_gold.jsonl"
    shutil.copy2(GOLD, gold_copy)
    print(f"📄 Copied gold -> {gold_copy}")

    # Initialize your pipeline (this loads retrieval config, T5 model, and chunks)
    print("🔧 Initializing pipeline…")
    load_retrieval_config()
    initialize_models()
    load_initial_data()

    # Generate predictions
    preds_path = out_dir / "preds.jsonl"
    n = 0
    with GOLD.open("r", encoding="utf-8") as fin, preds_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = rec.get("id")
            q   = rec.get("query", "")
            ans, _sources, retrieved_ids = _answer_question(q)
            fout.write(json.dumps({
                "id": qid,
                "query": q,
                "model_answer": ans,
                "retrieved_ids": retrieved_ids
            }, ensure_ascii=False) + "\n")
            n += 1

    print(f"✅ Wrote {n} predictions -> {preds_path}")

    # Optional scoring (off by default)
    if args.score:
        if not EVAL.exists():
            print("⚠️ evaluator.py not found; skipping scoring.")
        else:
            print("📊 Running evaluator.py…")
            subprocess.check_call([sys.executable, str(EVAL), "--output-dir", str(out_dir)])
            print(f"📁 Scoring output in: {out_dir}")

    print("\nDone. Context report folder:")
    print(f"  {out_dir}")
    print(f"   ├─ context_gold.jsonl")
    print(f"   └─ preds.jsonl")
    if args.score:
        print(f"   └─ report.json (if produced)")


if __name__ == "__main__":
    main()