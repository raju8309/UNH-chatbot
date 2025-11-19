import csv
import json
import os
import threading
from datetime import datetime
from typing import List, Optional

CHAT_LOG_PATH = "chat_logs.csv"
SELECTION_LOG_PATH = "answer_selections.csv"
_LOG_LOCK = threading.Lock()

def ensure_chat_log_file() -> None:
    """Initialize log files with headers if they don't exist."""
    if not os.path.isfile(CHAT_LOG_PATH):
        with open(CHAT_LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "question", "answer", "sources_json"])
        print(f"✓ Created chat log file: {CHAT_LOG_PATH}")
    
    # check if selection log needs header update
    needs_header_update = False
    if os.path.isfile(SELECTION_LOG_PATH):
        with open(SELECTION_LOG_PATH, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header and len(header) < 8:  # Old format
                needs_header_update = True
                print("Detected old answer_selections.csv format, will append new columns")
    
    if not os.path.isfile(SELECTION_LOG_PATH) or needs_header_update:
        # if file doesn't exist, create with new header
        if not os.path.isfile(SELECTION_LOG_PATH):
            with open(SELECTION_LOG_PATH, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "session_id", 
                    "selected_version", 
                    "answer_mode",
                    "question",
                    "primary_answer",
                    "alternative_answer",
                    "gold_similarity"
                ])
            print(f"Created answer selection log file: {SELECTION_LOG_PATH}")
        else:
            # file exists with old format - backup and recreate
            backup_path = SELECTION_LOG_PATH + ".backup"
            if not os.path.exists(backup_path):
                os.rename(SELECTION_LOG_PATH, backup_path)
                print(f"✓ Backed up old log to: {backup_path}")
            
            # create new file with proper header
            with open(SELECTION_LOG_PATH, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "session_id", 
                    "selected_version", 
                    "answer_mode",
                    "question",
                    "primary_answer",
                    "alternative_answer",
                    "gold_similarity"
                ])
            print(f"Created new answer selection log with enhanced columns")

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

def log_answer_selection(
    session_id: str, 
    selected_version: str, 
    answer_mode: str, 
    timestamp: str,
    question: Optional[str] = None,
    primary_answer: Optional[str] = None,
    alternative_answer: Optional[str] = None,
    gold_similarity: Optional[float] = None
) -> None:
    # clean up text fields to avoid CSV issues
    def clean_text(text):
        if text is None:
            return ""
        # replace newlines with spaces to keep CSV clean
        return str(text).replace('\n', ' ').replace('\r', ' ')
    
    row = [
        timestamp,
        session_id,
        selected_version,
        answer_mode,
        clean_text(question),
        clean_text(primary_answer),
        clean_text(alternative_answer),
        f"{gold_similarity:.4f}" if gold_similarity is not None else ""
    ]
    
    with _LOG_LOCK:
        with open(SELECTION_LOG_PATH, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)