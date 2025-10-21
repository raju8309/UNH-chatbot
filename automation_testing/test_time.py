import sys
import argparse
import json
import time, statistics
from pathlib import Path
import random
import asyncio

ROOT   = Path(__file__).resolve().parents[1]
GOLD   = ROOT / "automation_testing" / "gold.jsonl"

sys.path.insert(0, str(ROOT / "backend"))
from config.settings import load_retrieval_config
from models.ml_models import initialize_models
from services.chunk_service import load_initial_data
from routers.chat import answer_question
from models import ChatRequest

async def run_timing():
    """
    Measure time to answer a question using the chat API endpoint. Questions are randomly picked from the gold set.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=20, help="number of measured runs (after warmup)")
    args = ap.parse_args()

    # Load catalog
    load_retrieval_config()
    initialize_models()
    load_initial_data()

    # Load all queries from gold.jsonl
    queries = []
    with open(GOLD, "r", encoding="utf-8") as fin:
        for line in fin:
            if line.strip():
                rec = json.loads(line)
                queries.append(rec["query"])

    # Timed questions
    times = []
    for _ in range(args.runs):
        q = random.choice(queries)
        t0 = time.perf_counter()
        _ = await answer_question(ChatRequest(message=q))
        times.append(time.perf_counter() - t0)

    # Results
    times.sort()
    p50 = statistics.median(times)
    p95 = times[int(0.95 * len(times)) - 1]
    p99 = times[int(0.99 * len(times)) - 1]
    print({"runs": args.runs, "p50": p50, "p95": p95, "p99": p99})

if __name__ == "__main__":
    asyncio.run(run_timing())
