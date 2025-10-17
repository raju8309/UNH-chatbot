import csv
import json
import os
import threading
from datetime import datetime
from typing import List

CHAT_LOG_PATH = "chat_logs.csv"
_LOG_LOCK = threading.Lock()

def ensure_chat_log_file() -> None:
    if not os.path.isfile(CHAT_LOG_PATH):
        with open(CHAT_LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "question", "answer", "sources_json"])
        print(f"✓ Created chat log file: {CHAT_LOG_PATH}")

def log_chat_interaction(question: str, answer: str, sources: List[str]) -> None:
    timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    row = [
        timestamp,
        question,
        answer,
        json.dumps(sources, ensure_ascii=False)
    ]
    
    with _LOG_LOCK:
        with open(CHAT_LOG_PATH, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)