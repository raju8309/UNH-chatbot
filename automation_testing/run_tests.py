#!/usr/bin/env python3
import sys, subprocess, json
from pathlib import Path

ROOT   = Path(__file__).resolve().parents[1]
PY     = sys.executable
PRED   = ROOT / "automation_testing" / "predict.py"
EVAL   = ROOT / "automation_testing" / "evaluator.py"
GOLD   = ROOT / "automation_testing" / "gold.jsonl"
REPORT = ROOT / "automation_testing" / "report.json"

def run(cmd):
    print("→", " ".join(str(c) for c in cmd))
    subprocess.check_call(cmd)

def main():
    if not GOLD.exists():
        raise SystemExit(f"Missing gold file: {GOLD}")
    if not PRED.exists():
        raise SystemExit(f"Missing predictor: {PRED}")
    if not EVAL.exists():
        raise SystemExit(f"Missing evaluator: {EVAL}")

    try:
        run([PY, str(PRED), "--offline"])
    except subprocess.CalledProcessError:
        run([PY, str(PRED)])

  
    run([PY, str(EVAL)])

    
    if REPORT.exists():
        try:
            data = json.loads(REPORT.read_text())
            print("\n=== Summary ===")
            print(json.dumps(data.get("summary", data), indent=2))
        except Exception as e:
            print(f"(Could not read summary: {e})")

    print("\n Done. Outputs: automation_testing/preds.jsonl and automation_testing/report.json")

if __name__ == "__main__":
    main()
