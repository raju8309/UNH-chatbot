from pathlib import Path
from typing import Any, Dict, Tuple
import yaml

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
    
    # set defaults for missing keys
    if "policy_terms" not in CFG:
        CFG["policy_terms"] = []
    
    if "tier_boosts" not in CFG:
        CFG["tier_boosts"] = {1: 1.35, 2: 1.10, 3: 1.0, 4: 1.0}
    
    if "intent" not in CFG:
        CFG["intent"] = {
            "course_keywords": [],
            "degree_keywords": [],
            "course_code_regex": r"\b[A-Z]{3,5}\s?\d{3}\b",
        }
    
    if "nudges" not in CFG:
        CFG["nudges"] = {"policy_acadreg_url": 1.15}
    
    if "guarantees" not in CFG:
        CFG["guarantees"] = {
            "ensure_tier1_on_policy": True,
            "ensure_tier4_on_program": True
        }
    
    if "tier4_gate" not in CFG:
        CFG["tier4_gate"] = {
            "use_embedding": True,
            "min_title_sim": 0.42,
            "min_alt_sim": 0.38
        }
    
    POLICY_TERMS = tuple(CFG.get("policy_terms", []))
    print("Configuration loaded")

def get_config() -> Dict[str, Any]:
    return CFG

def get_policy_terms() -> Tuple[str, ...]:
    return POLICY_TERMS