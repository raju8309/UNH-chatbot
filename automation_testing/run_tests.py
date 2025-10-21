#!/usr/bin/env python3
import sys, subprocess, json
import shutil
from pathlib import Path
from datetime import datetime
import asyncio

ROOT   = Path(__file__).resolve().parents[1]
PY     = sys.executable
EVAL   = ROOT / "automation_testing" / "evaluator.py"
GOLD   = ROOT / "automation_testing" / "gold.jsonl"

sys.path.insert(0, str(ROOT / "backend"))

from config.settings import load_retrieval_config
from models.ml_models import initialize_models
from services.chunk_service import load_initial_data
from services.session_service import clear_all_sessions
from routers.chat import answer_question
from models.api_models import ChatRequest

def run(cmd):
    print("→", " ".join(str(c) for c in cmd))
    subprocess.check_call(cmd)

async def main():
    if not GOLD.exists():
        raise SystemExit(f"Missing gold file: {GOLD}")
    if not EVAL.exists():
        raise SystemExit(f"Missing evaluator: {EVAL}")

    # Create timestamped report directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    reports_dir = ROOT / "automation_testing" / "reports"
    report_dir = reports_dir / f"{timestamp}"
    report_dir.mkdir(parents=True, exist_ok=True)

    # Copy gold.jsonl to the test run directory
    gold_copy = report_dir / "gold.jsonl"
    shutil.copy2(GOLD, gold_copy)
    print(f"Copied {GOLD} to {gold_copy}")

    # Load catalog
    load_retrieval_config()
    initialize_models()
    load_initial_data()

    # Generate predictions using the real pipeline
    preds_path = report_dir / "preds.jsonl"
    with open(GOLD, "r", encoding="utf-8") as fin, open(preds_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            rec = json.loads(line)
            qid = rec["id"]
            q   = rec["query"]

            response = await answer_question(ChatRequest(message=q))
            clear_all_sessions()
            
            fout.write(json.dumps({
                "id": qid,
                "model_answer": response.answer,
                "retrieved_ids": response.retrieval_path
            }, ensure_ascii=False) + "\n")

    print(f"Wrote predictions to {preds_path}")


    run([PY, str(EVAL), "--output-dir", str(report_dir)])

    report_file = report_dir / "report.json"
    if report_file.exists():
        try:
            data = json.loads(report_file.read_text())
            print("\n=== Summary ===")
            print(json.dumps(data.get("summary", data), indent=2))
        except Exception as e:
            print(f"(Could not read summary: {e})")

    print(f"\n Done. Outputs in: {report_dir}")
    print(f" - {report_dir / 'gold.jsonl'} (copy of test data)")
    print(f" - {report_dir / 'preds.jsonl'}")
    print(f" - {report_dir / 'report.json'}")

if __name__ == "__main__":
    asyncio.run(main())