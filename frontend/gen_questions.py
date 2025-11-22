#!/usr/bin/env python3
"""
Generate popular questions for the frontend from the gold set only.
This script pulls all questions from automation_testing/gold.jsonl
and saves them to frontend/public/popular_questions.json
"""

import json
from pathlib import Path

# determine base directory automatically
# in Docker /app exists, locally use repo root 
if Path("/app").exists():
    BASE_DIR = Path("/app")
else:
    BASE_DIR = Path(__file__).resolve().parents[1]

# file paths
GOLD_FILE = BASE_DIR / "automation_testing" / "gold.jsonl"
OUTPUT_FILE = BASE_DIR / "frontend" / "public" / "popular_questions.json"

# load gold set questions
gold_questions = []
try:
    with open(GOLD_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                query = entry.get("query", "").strip()
                if query:
                    gold_questions.append(query)
    
    print(f"Loaded {len(gold_questions)} questions from gold set")
    
except FileNotFoundError:
    print(f"[ERROR] GOLD_FILE not found: {GOLD_FILE}")
    print("Cannot generate popular questions without gold set")
    exit(1)

# save to output file
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump({"questions": gold_questions}, f, indent=2, ensure_ascii=False)

print(f"Saved {len(gold_questions)} questions to {OUTPUT_FILE}")