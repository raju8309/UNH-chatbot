from pathlib import Path
from typing import Any, Dict, Tuple
import yaml
import os

# global configuration
CFG: Dict[str, Any] = {}
POLICY_TERMS: Tuple[str, ...] = ()

def load_retrieval_config() -> None:
    global CFG, POLICY_TERMS
    cfg_path = Path(__file__).resolve().parent / "retrieval.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            CFG = yaml.safe_load(f) or {}
    else:
        CFG = {}

    CFG.setdefault("policy_terms", [])
    CFG.setdefault("tier_system", {"enabled": True})
    CFG.setdefault("tier_boosts", {0: 3.0, 1: 1.25, 2: 1.10, 3: 1.0, 4: 1.0})
    CFG.setdefault("intent", {
        "course_keywords": [],
        "degree_keywords": [],
        "course_code_regex": r"\b[A-Z]{3,5}\s?\d{3}\b",
    })
    CFG.setdefault("nudges", {"policy_acadreg_url": 1.15})
    CFG.setdefault("guarantees", {"ensure_tier1_on_policy": True, "ensure_tier4_on_program": True})
    CFG.setdefault("tier4_gate", {"use_embedding": True, "min_title_sim": 0.42, "min_alt_sim": 0.38})

    rs = CFG.get("retrieval_sizes", {})
    CFG["search"] = {
        "topn_default": int(rs.get("topn_default", 40))
    }
    CFG["k"] = int(rs.get("k", 5))

    gold_cfg = CFG.setdefault("gold_set", {})
    gold_cfg.setdefault("enabled", False)  # <-- ensure gold set disabled by default
    gold_cfg.setdefault("gold_file_path", "../automation_testing/gold.jsonl")
    gold_cfg.setdefault("enable_direct_answer", True)
    gold_cfg.setdefault("direct_answer_threshold", 0.95)
    gold_cfg.setdefault("ensure_gold_in_results", True)

    CFG.setdefault("calendar_linking", {"enabled": True})

    POLICY_TERMS = tuple(CFG.get("policy_terms", []))
    print("Configuration loaded")
    print("Gold set enabled?", gold_cfg["enabled"])

def get_config() -> Dict[str, Any]:
    return CFG

def get_policy_terms() -> Tuple[str, ...]:
    return POLICY_TERMS

# Embedding model for retrieval (NOT the answer LLM)
# Default stays MiniLM; can override via env var.
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# --- Query Transformation ---
ENABLE_QUERY_REWRITER = True  # set False to disable LLM rewrite; rules still apply

# Forbidden rewrites and constraints
FORBIDDEN_PHRASES = {"external graduate transfer credits"}
REWRITE_MIN_WORDS = 5
REWRITE_MAX_WORDS = 25

# LLM rewrite sampling and semantic similarity guardrails
# Number of candidate rewrites to sample before picking the best one
REWRITE_NUM_CANDIDATES = int(os.getenv("REWRITE_NUM_CANDIDATES", "3"))
# Enable semantic similarity check between original and rewritten query
REWRITE_USE_SEMANTIC_SIMILARITY = True
# Minimum cosine similarity required to accept a rewrite when similarity checks are enabled
REWRITE_MIN_SIMILARITY = float(os.getenv("REWRITE_MIN_SIMILARITY", "0.6"))

# Retrieval options
RETRIEVAL_USE_DUAL_QUERY = True
