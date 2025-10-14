import csv
import json
from collections import Counter

INPUT_FILE = "../chat_logs.csv"         # your raw question log
GOLD_FILE = "../automation_testing/gold.jsonl"   # your gold set JSONL
OUTPUT_FILE = "public/popular_questions.json"
TOP_N = 6

# load saved questions from chat_logs.csv
try:
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        existing_data = json.load(f)
        existing_questions = set(existing_data.get("questions", []))
except FileNotFoundError:
    existing_questions = set()

# load gold set questions
gold_questions = set()
with open(GOLD_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            entry = json.loads(line)
            query = entry.get("query", "").strip()
            if query:
                gold_questions.add(query)

# read and count questions from csv
with open(INPUT_FILE, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    csv_questions = [row['question'].strip() for row in reader if row.get('question')]

counts = Counter(csv_questions)
top_csv_questions = [q for q, _ in counts.most_common(TOP_N)]

# merge questions, existing + gold + new top CSV questions
all_new_questions = [q for q in top_csv_questions if q not in existing_questions]
updated_questions = list(existing_questions | gold_questions | set(all_new_questions))

# save updated popular questions
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump({"questions": updated_questions}, f, indent=2, ensure_ascii=False)

print(f"Existing questions: {len(existing_questions)}")
print(f"Gold questions added: {len(gold_questions - existing_questions)}")
print(f"New CSV top questions added: {len(all_new_questions)}")
print(f"Total questions now in {OUTPUT_FILE}: {len(updated_questions)}")
