#!/usr/bin/env python3
"""
Evaluation script for QA predictions against a gold set.

Metrics:
- Nugget precision / recall / F1 using SBERT similarity
- SBERT cosine similarity to reference answer
- BERTScore F1 to reference answer
- Retrieval Recall@k and nDCG@k (k in {1,3,5})

Outputs:
- report.json with per-question rows and an aggregate summary
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
from bert_score import score as bertscore
from sentence_transformers import SentenceTransformer, util

# -------- Paths --------
GOLD = Path(__file__).with_name("gold.jsonl")
PREDS = Path(__file__).with_name("preds.jsonl")
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
    def sbert_cosine(self, answer: str, reference: str) -> float:
        """Cosine similarity between answer and reference, using SBERT."""
        em = self.model.encode(
            [answer, reference],
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        sim = util.cos_sim(em[0], em[1]).cpu().numpy()
        return float(sim)

    def bertscore_f1(self, reference: str, answer: str) -> float:
        """BERTScore F1 between answer and reference (English, with baseline)."""
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
        if not nuggets:
            # If there are no nuggets to hit, treat as perfect coverage.
            return (1.0, 1.0, 1.0)

        a = self.model.encode(
            [answer],
            convert_to_tensor=True,
            normalize_embeddings=True,
        )[0]
        ns = self.model.encode(
            nuggets,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        sims = util.cos_sim(ns, a).cpu().numpy().flatten()

        tp = int((sims >= self.thresh).sum())
        fn = len(nuggets) - tp
        fp = 0  # we only score nuggets -> no negative labels tracked

        precision = tp / (tp + fp) if (tp + fp) else 1.0
        recall = tp / (tp + fn) if (tp + fn) else 1.0
        f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)
        return precision, recall, f1


# -------- Main --------
def main() -> None:
    # Load gold and predictions keyed by id
    gold: Dict[str, Dict] = {r["id"]: r for r in read_jsonl(GOLD)}
    preds: Dict[str, Dict] = {r["id"]: r for r in read_jsonl(PREDS)}

    ev = Evaluator()

    # Buckets to accumulate metrics
    per_question: List[Dict] = []
    buckets = {k: [] for k in ["nugP", "nugR", "nugF1", "sbert", "bsF1", "R@1", "R@3", "R@5", "N@1", "N@3", "N@5"]}

    for qid, g in gold.items():
        p = preds.get(qid, {"model_answer": "", "retrieved_ids": []})
        answer = p["model_answer"]
        ref = g.get("reference_answer", "")
        nuggets = g.get("nuggets", [])
        gold_passages = set(g.get("gold_passages", []))
        retrieved = p["retrieved_ids"]

        # Content metrics
        nugP, nugR, nugF1 = ev.nugget_prf(nuggets, answer)
        sbert = ev.sbert_cosine(answer, ref)
        bsF1 = ev.bertscore_f1(ref, answer)

        # Retrieval metrics
        r1, r3, r5 = (recall_at_k(retrieved, gold_passages, k) for k in (1, 3, 5))
        n1, n3, n5 = (ndcg_at_k(retrieved, gold_passages, k) for k in (1, 3, 5))

        row = {
            "id": qid,
            "nugget_precision": nugP,
            "nugget_recall": nugR,
            "nugget_f1": nugF1,
            "sbert_cosine": sbert,
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
            ["nugP", "nugR", "nugF1", "sbert", "bsF1", "R@1", "R@3", "R@5", "N@1", "N@3", "N@5"],
            [nugP, nugR, nugF1, sbert, bsF1, r1, r3, r5, n1, n3, n5],
        ):
            buckets[key].append(val)

    mean = lambda xs: float(np.mean(xs)) if xs else 0.0  # noqa: E731

    summary = {
        "count": len(per_question),
        "nugget_precision": mean(buckets["nugP"]),
        "nugget_recall": mean(buckets["nugR"]),
        "nugget_f1": mean(buckets["nugF1"]),
        "sbert_cosine": mean(buckets["sbert"]),
        "bertscore_f1": mean(buckets["bsF1"]),
        "recall@1": mean(buckets["R@1"]),
        "recall@3": mean(buckets["R@3"]),
        "recall@5": mean(buckets["R@5"]),
        "ndcg@1": mean(buckets["N@1"]),
        "ndcg@3": mean(buckets["N@3"]),
        "ndcg@5": mean(buckets["N@5"]),
    }

    REPORT.write_text(
        json.dumps({"per_question": per_question, "summary": summary}, indent=2),
        encoding="utf-8",
    )
    print("Wrote", REPORT)


if __name__ == "__main__":
    main()
