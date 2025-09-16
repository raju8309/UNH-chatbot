# automation_testing/run_tests.py
import csv
import importlib
import sys
from pathlib import Path

# --- Locate repo root (parent of automation_testing) and add to sys.path ---
HERE = Path(__file__).resolve()
ROOT = HERE.parent.parent            # .../Fall2025-Team-Goopy
sys.path.insert(0, str(ROOT))

# --- Resolve CSV path (arg or default) ---
DEFAULTS = [
    ROOT / "automation_testing" / "tests.csv",
    ROOT / "tests.csv",
]
CSV_PATH = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else next(
    (p for p in DEFAULTS if p.exists()),
    DEFAULTS[0]
)

print("Loading app…")
app = importlib.import_module("main")  # imports ROOT/main.py

# Build embeddings from JSON (assumes file lives at repo root)
JSON_PATH = ROOT / "../scrape/degree_requirements.json"
app.load_json_file(str(JSON_PATH))
JSON_PATH = ROOT / "../scape/course_descriptions.json"
app.load_json_file(str(JSON_PATH))

def ask(q: str) -> str:
    out = app.answer_question(q)
    return (out or "").strip()

def passed(answer: str, any_of_bits):
    ans_lower = answer.lower()
    return any(bit.strip().lower() in ans_lower for bit in any_of_bits)

total = 0
ok = 0
fail_rows = []

print(f"Running tests from: {CSV_PATH}\n")
with open(CSV_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        total += 1
        q = row["question"].strip()
        bits = [b for b in row["expected_any_of"].split("|") if b.strip()]
        a = ask(q)
        is_ok = passed(a, bits)
        status = "PASS" if is_ok else "FAIL"
        if is_ok:
            ok += 1
        else:
            fail_rows.append((q, a))
        print(f"[{status}] {q}\n→ {a}\n")

print("=" * 60)
print(f"TOTAL: {ok}/{total} passed")

# Non-zero exit code if any failures (useful for CI)
if fail_rows:
    print("\nFailed cases:")
    for q, a in fail_rows:
        print(f"- Q: {q}\n  A: {a}\n")
    sys.exit(1)
sys.exit(0)
