#!/usr/bin/env python3
"""
Evaluation script for QA predictions against a gold set.

Metrics:
- Nugget precision / recall / F1 using SBERT similarity
- SBERT cosine similarity to reference answer
- SBERT cosine similarity to top retrieved chunk text
- BERTScore F1 to reference answer
- Retrieval Recall@k and nDCG@k (k in {1,3,5})

Outputs:
- report.json with per-question rows and an aggregate summary
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple
from urllib.parse import urlparse

import numpy as np
from bert_score import score as bertscore
from sentence_transformers import SentenceTransformer, util

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

# -------- Default file locations (when no --output-dir is given) -------
GOLD   = Path(__file__).with_name("gold.jsonl")
PREDS  = Path(__file__).with_name("preds.jsonl")
REPORT = Path(__file__).with_name("report.json")


# -------- I/O helpers --------
def read_jsonl(path: Path) -> Iterable[Dict]:
    """Yield JSON objects from a .jsonl file (one JSON per line)."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# -------- Retrieval helpers --------
def base_id(cid: str) -> str:
    """Strip chunk suffix after '#', leaving the base document id."""
    return (cid or "").split("#", 1)[0]


def dcg(relevances: List[int]) -> float:
    """Discounted Cumulative Gain."""
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_k(pred_ids: List[str], gold_ids: Set[str], k: int) -> float:
    """nDCG@k using binary relevance."""
    top = pred_ids[:k]
    relevances = [1 if base_id(x) in gold_ids else 0 for x in top]
    ideal = dcg(sorted(relevances, reverse=True))
    return 0.0 if ideal == 0 else dcg(relevances) / ideal


def recall_at_k(pred_ids: List[str], gold_ids: Set[str], k: int) -> float:
    """Recall@k on base doc ids."""
    if not gold_ids:
        return 0.0
    preds = {base_id(x) for x in pred_ids[:k]}
    return len(preds & gold_ids) / len(gold_ids)


# -------- Evaluator --------
class Evaluator:
    """
    Wraps sentence-transformers for similarity-based metrics.

    Parameters
    ----------
    thresh : float
        Similarity threshold used to count a nugget as matched.
    model_name : str
        Sentence-Transformer model used for embeddings.
    """

    def __init__(
        self,
        thresh: float = 0.70,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.thresh = thresh
        self.model = SentenceTransformer(model_name)

    # --- Answer-to-reference metrics ---
    def sbert_cosine(self, a: str, b: str) -> float:
        """Cosine similarity between two texts, using SBERT."""
        a = (a or "").strip()
        b = (b or "").strip()
        if not a or not b:
            return 0.0
        em = self.model.encode(
            [a, b],
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        sim = util.cos_sim(em[0], em[1]).cpu().numpy()
        return float(sim)

    def bertscore_f1(self, reference: str, answer: str) -> float:
        """BERTScore F1 between answer and reference (English, with baseline)."""
        reference = (reference or "").strip()
        answer = (answer or "").strip()
        if not reference or not answer:
            return 0.0
        _, _, f1 = bertscore(
            [answer],
            [reference],
            lang="en",
            rescale_with_baseline=True,
            verbose=False,
        )
        return float(f1[0].item())

    # --- Nugget coverage metrics ---
    def nugget_prf(self, nuggets: List[str], answer: str) -> Tuple[float, float, float]:
        """
        Compute precision / recall / F1 of nuggets covered by the answer.
        A nugget counts as covered if its cosine similarity to the answer
        exceeds the threshold.
        """
        nuggets = [n.strip() for n in (nuggets or []) if isinstance(n, str) and n.strip()]
        answer = (answer or "").strip()

        if not nuggets:
            # No nuggets to hit -> perfect by convention
            return (1.0, 1.0, 1.0)
        if not answer:
            return (1.0, 0.0, 0.0)  # precision 1.0 (no predicted negatives), recall 0.0

        a = self.model.encode([answer], convert_to_tensor=True, normalize_embeddings=True)[0]
        ns = self.model.encode(nuggets, convert_to_tensor=True, normalize_embeddings=True)
        sims = util.cos_sim(ns, a).cpu().numpy().flatten()

        tp = int((sims >= self.thresh).sum())
        fn = len(nuggets) - tp
        fp = 0  # we only score nuggets -> no negative labels tracked

        precision = tp / (tp + fp) if (tp + fp) else 1.0
        recall = tp / (tp + fn) if (tp + fn) else 1.0
        f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)
        return precision, recall, f1


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=str, help="Directory to read preds.jsonl from and write report.json to")
    return ap.parse_args()

def _base_doc_id(url: str) -> str:
    if not url:
        return "catalog"
    p = urlparse(url)
    name = (Path(p.path).name or "").rstrip("/")
    if not name and p.path:
        name = Path(p.path).parts[-1]
    slug = (name or "catalog").replace(".html", "").replace(".htm", "") or "catalog"
    return slug

def auto_find_latest_reports_dir() -> Path | None:
    """Pick the newest automation_testing/reports/<timestamp> that has preds.jsonl."""
    reports_root = Path(__file__).with_name("reports")
    if not reports_root.exists():
        return None
    candidates = sorted(
        (p for p in reports_root.iterdir() if p.is_dir()),
        key=lambda p: p.name,
        reverse=True,
    )
    for c in candidates:
        if (c / "preds.jsonl").exists():
            return c
    return None


# -------- Main --------
if __name__ == "__main__":
    args = parse_args()

    # Decide where to read/write
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # If no output dir passed, auto-discover the latest reports run
        output_dir = auto_find_latest_reports_dir()
        if output_dir is None:
            # Fall back to flat files next to this script (legacy mode)
            output_dir = None

    if output_dir is not None:
        preds_path = output_dir / "preds.jsonl"
        report_path = output_dir / "report.json"
        if not preds_path.exists():
            raise SystemExit(f"Could not find predictions file: {preds_path}\n"
                             f"Run run_tests.py first, or pass --output-dir to evaluator.py.")
        print(f"Using run directory: {output_dir}")
    else:
        preds_path = PREDS
        report_path = REPORT
        if not preds_path.exists():
            raise SystemExit(
                f"Could not find {preds_path}. "
                f"Run run_tests.py or pass --output-dir pointing to automation_testing/reports/<timestamp>."
            )
        print(f"Using flat files next to evaluator: {preds_path}")

    if not GOLD.exists():
        raise SystemExit(f"Gold file not found: {GOLD}")

    # Load gold and predictions keyed by id
    gold: Dict[str, Dict] = {r["id"]: r for r in read_jsonl(GOLD)}
    preds: Dict[str, Dict] = {r["id"]: r for r in read_jsonl(preds_path)}

    ev = Evaluator()

    # Buckets to accumulate metrics
    per_question: List[Dict] = []
    buckets = {k: [] for k in [
        "nugP", "nugR", "nugF1", "sbert", "sbert_chunk", "bsF1",
        "R@1", "R@3", "R@5", "N@1", "N@3", "N@5"
    ]}

    for qid, g in gold.items():
        p = preds.get(qid, {"model_answer": "", "retrieved_ids": []})
        answer = (p.get("model_answer") or "").strip()
        ref = (g.get("reference_answer") or "").strip()
        nuggets = g.get("nuggets", [])
        gold_passages = set(g.get("gold_passages", []))
        retrieved = p.get("retrieved_ids", []) or []

        # --- Grab any available retrieved chunk texts for sbert_cosine_chunk (best of top-3) 
        retrieved_texts: List[str] = []
        if isinstance(p.get("retrieved_ids"), list):
            for r in p["retrieved_ids"][:3]:
                if isinstance(r, dict):
                    t = r.get("text")
                    if isinstance(t, str) and t.strip():
                        retrieved_texts.append(t.strip())

        # Normalize retrieved ids for recall/ndcg
        if retrieved and isinstance(retrieved, list) and len(retrieved) > 0 and isinstance(retrieved[0], dict):
            retrieved = [
                f"{_base_doc_id(item.get('url', ''))}#{item.get('idx', '')}"
                for item in retrieved if isinstance(item, dict)
            ]
        elif not isinstance(retrieved, list):
            retrieved = []

        # Content metrics
        nugP, nugR, nugF1 = ev.nugget_prf(nuggets, answer)
        sbert = ev.sbert_cosine(answer, ref)
        bsF1 = ev.bertscore_f1(ref, answer)
        
        if retrieved_texts:
            sbert_chunk = max(ev.sbert_cosine(txt, ref) for txt in retrieved_texts)
        else:
            sbert_chunk = 0.0

        # Retrieval metrics
        r1, r3, r5 = (recall_at_k(retrieved, gold_passages, k) for k in (1, 3, 5))
        n1, n3, n5 = (ndcg_at_k(retrieved, gold_passages, k) for k in (1, 3, 5))

        row = {
            "id": qid,
            "nugget_precision": nugP,
            "nugget_recall": nugR,
            "nugget_f1": nugF1,
            "sbert_cosine": sbert,
            "sbert_cosine_chunk": sbert_chunk,
            "bertscore_f1": bsF1,
            "recall@1": r1,
            "recall@3": r3,
            "recall@5": r5,
            "ndcg@1": n1,
            "ndcg@3": n3,
            "ndcg@5": n5,
        }
        per_question.append(row)

        for key, val in zip(
            ["nugP", "nugR", "nugF1", "sbert", "sbert_chunk", "bsF1", "R@1", "R@3", "R@5", "N@1", "N@3", "N@5"],
            [nugP, nugR, nugF1, sbert, sbert_chunk, bsF1, r1, r3, r5, n1, n3, n5],
        ):
            buckets[key].append(val)

    mean = lambda xs: float(np.mean(xs)) if xs else 0.0

    summary = {
        "count": len(per_question),
        "nugget_precision": mean(buckets["nugP"]),
        "nugget_recall": mean(buckets["nugR"]),
        "nugget_f1": mean(buckets["nugF1"]),
        "sbert_cosine": mean(buckets["sbert"]),
        "sbert_cosine_chunk": mean(buckets["sbert_chunk"]),
        "bertscore_f1": mean(buckets["bsF1"]),
        "recall@1": mean(buckets["R@1"]),
        "recall@3": mean(buckets["R@3"]),
        "recall@5": mean(buckets["R@5"]),
        "ndcg@1": mean(buckets["N@1"]),
        "ndcg@3": mean(buckets["N@3"]),
        "ndcg@5": mean(buckets["N@5"]),
    }

    report_path.write_text(
        json.dumps({"per_question": per_question, "summary": summary}, indent=2),
        encoding="utf-8",
    )
    print("Wrote", report_path)