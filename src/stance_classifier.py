"""
Neuro-Symbolic Stance & Contradiction Detection Engine
------------------------------------------------------
Evaluates cross-source claim pairs within each cluster using:
1. Deterministic Neuro-Symbolic Rules (exact match, numerical disparity, unit validation, temporal progression)
2. Groq LLM Chain-of-Thought (CoT) Classifier for linguistic nuances and semantic intent.

Stance Labels (5-Class):
- AGREE
- DISAGREE
- DISCUSS
- UNRELATED
- INSUFFICIENT_EVIDENCE
"""

import sys
import os
import re
import json
import itertools
import time
from typing import List, Dict, Any, Optional, Tuple
from groq import Groq

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    get_groq_api_key,
    GROQ_MODEL,
    STANCE_MAX_TOKENS,
    CLUSTERS_FILE,
    CLAIMS_FILE,
)
from src.clustering import cluster_claims

COT_STANCE_PROMPT = """You are a precision stance classifier following the FNC-1 (Fake News Challenge) task definition.
Analyze the factual relationship between Claim B and Claim A regarding the same incident or situation.

Possible Stance Categories:
- "AGREE": Claim B confirms, supports, or is substantively consistent with Claim A (same facts, corroborated counts, or same status).
- "DISAGREE": Claim B directly contradicts Claim A (incompatible numbers, conflicting locations, opposite outcomes, or conflicting official accounts for the same event window).
- "DISCUSS": Claim B discusses the same topic but neither strictly confirms nor refutes (e.g. chronological progression across different dates, or supplementary details).
- "UNRELATED": Claim B discusses an entirely different event, entity, or metric.
- "INSUFFICIENT_EVIDENCE": The claims are vague, lack temporal/geographical anchors, or lack enough evidence to confirm or contradict.

Claim A (Source: {source_a}, Date: {date_a}):
"{claim_a}"

Claim B (Source: {source_b}, Date: {date_b}):
"{claim_b}"

Guidelines:
1. Compare facts, numerical figures, and reporting dates step-by-step.
2. If numbers differ because one report is from an earlier date and another is from a later date during an unfolding event, classify as "DISCUSS" (temporal update), NOT "DISAGREE".
3. If numbers differ substantially for the same time window or are un-reconciled, classify as "DISAGREE".
4. Output valid JSON only with keys: "stance", "conflicting_attribute", "reasoning", "confidence".

JSON Output Format:
{{
  "stance": "AGREE | DISAGREE | DISCUSS | UNRELATED | INSUFFICIENT_EVIDENCE",
  "conflicting_attribute": "e.g. Death Toll, Affected Population, Location, or null",
  "reasoning": "1-2 sentence comparison explaining the verdict.",
  "confidence": 0.95
}}
"""


def deterministic_stance_check(claim_a: Dict[str, Any], claim_b: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Applies high-precision deterministic rules based on numbers, units, topics, and dates.
    Returns stance dictionary if a rule fires, else None (delegating to LLM).
    """
    text_a = claim_a.get("normalized_claim") or claim_a.get("claim", "")
    text_b = claim_b.get("normalized_claim") or claim_b.get("claim", "")

    # Rule 0: Exact or normalized text match
    norm_a = re.sub(r"[^\w\s]", "", text_a).strip().lower()
    norm_b = re.sub(r"[^\w\s]", "", text_b).strip().lower()
    if norm_a == norm_b and len(norm_a) > 5:
        return {
            "stance": "AGREE",
            "conflicting_attribute": None,
            "reasoning": "Identical factual assertion reported across independent outlets.",
            "confidence": 0.99,
        }

    # Rule 1: Divergent topic categories
    top_a = claim_a.get("category") or claim_a.get("topic")
    top_b = claim_b.get("category") or claim_b.get("topic")
    if top_a and top_b and top_a != "general" and top_b != "general" and top_a != top_b:
        return {
            "stance": "UNRELATED",
            "conflicting_attribute": None,
            "reasoning": f"Different topical categories ({top_a} vs {top_b}).",
            "confidence": 0.95,
        }

    val_a = claim_a.get("numerical_value")
    val_b = claim_b.get("numerical_value")
    unit_a = str(claim_a.get("numerical_unit") or "").strip().lower()
    unit_b = str(claim_b.get("numerical_unit") or "").strip().lower()
    date_a = str(claim_a.get("date") or "").strip().lower()
    date_b = str(claim_b.get("date") or "").strip().lower()

    # Rule 2: Incompatible unit types (e.g. deaths vs injured are NOT contradictory)
    if unit_a and unit_b and unit_a != unit_b:
        incompatible_pairs = [("death", "injur"), ("dead", "injur"), ("people", "village"), ("district", "people")]
        for u1, u2 in incompatible_pairs:
            if (u1 in unit_a and u2 in unit_b) or (u2 in unit_a and u1 in unit_b):
                return {
                    "stance": "DISCUSS",
                    "conflicting_attribute": None,
                    "reasoning": f"Complementary metrics with differing units ({unit_a} vs {unit_b}), not a contradiction.",
                    "confidence": 0.90,
                }

    # Numerical Comparisons
    if val_a is not None and val_b is not None:
        try:
            num_a = float(val_a)
            num_b = float(val_b)
            if num_a > 0 and num_b > 0:
                ratio = max(num_a, num_b) / min(num_a, num_b)

                # Rule 3: Exact identical matching numbers on compatible units
                if ratio == 1.0 and (not unit_a or not unit_b or unit_a == unit_b):
                    return {
                        "stance": "AGREE",
                        "conflicting_attribute": None,
                        "reasoning": f"Exact matching statistics ({num_a:g}) confirmed by both outlets.",
                        "confidence": 0.98,
                    }

                # Rule 4: Chronological progression across distinct, identifiable dates
                if date_a and date_b and date_a != date_b and ratio >= 1.05:
                    return {
                        "stance": "DISCUSS",
                        "conflicting_attribute": None,
                        "reasoning": f"Temporal update: {claim_a.get('source_name')} reported {num_a:g} ({date_a}) while {claim_b.get('source_name')} reported {num_b:g} ({date_b}). Figures naturally evolved over time.",
                        "confidence": 0.92,
                    }

                # Rule 5: Significant numerical contradiction for overlapping / undated timeframe
                if (date_a == date_b or not date_a or not date_b) and ratio >= 1.25:
                    if not unit_a or not unit_b or unit_a == unit_b:
                        attribute = top_a.replace("_", " ").title() if top_a else "Numerical Metric"
                        return {
                            "stance": "DISAGREE",
                            "conflicting_attribute": attribute,
                            "reasoning": f"Severe numerical conflict: {claim_a.get('source_name')} reports {num_a:g} vs {claim_b.get('source_name')} reports {num_b:g} ({ratio:.1f}x disparity) for the same event window.",
                            "confidence": 0.96,
                        }
        except (ValueError, TypeError):
            pass

    return None


def classify_stance_pair(
    claim_a: Dict[str, Any],
    claim_b: Dict[str, Any],
    client: Optional[Groq] = None,
    max_retries: int = 3
) -> Dict[str, Any]:
    """
    Classifies the stance between two claims using deterministic rules first,
    falling back to Groq LLM with Chain-of-Thought prompting.
    """
    # 1. Deterministic Rule Check
    rule_result = deterministic_stance_check(claim_a, claim_b)
    if rule_result is not None:
        return rule_result

    # 2. LLM Chain-of-Thought Fallback
    if client is None:
        api_key = get_groq_api_key()
        if not api_key:
            return {
                "stance": "DISCUSS",
                "conflicting_attribute": None,
                "reasoning": "Defaulted to discuss; API key not configured.",
                "confidence": 0.50,
            }
        client = Groq(api_key=api_key)

    prompt = COT_STANCE_PROMPT.format(
        claim_a=claim_a.get("claim", ""),
        source_a=claim_a.get("source_name", "Source A"),
        date_a=claim_a.get("date", "not specified"),
        claim_b=claim_b.get("claim", ""),
        source_b=claim_b.get("source_name", "Source B"),
        date_b=claim_b.get("date", "not specified"),
    )

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                temperature=0.0,
                max_tokens=STANCE_MAX_TOKENS,
                messages=[
                    {"role": "system", "content": "You are an objective stance and contradiction detector. Return valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            raw = response.choices[0].message.content.strip()
            data = json.loads(raw)

            stance_raw = str(data.get("stance", "DISCUSS")).strip().upper()
            valid_stances = ["AGREE", "DISAGREE", "DISCUSS", "UNRELATED", "INSUFFICIENT_EVIDENCE"]
            final_stance = stance_raw if stance_raw in valid_stances else "DISCUSS"

            return {
                "stance": final_stance,
                "conflicting_attribute": data.get("conflicting_attribute"),
                "reasoning": data.get("reasoning", "Cross-source factual stance verified by AI model."),
                "confidence": float(data.get("confidence", 0.85)),
            }
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "rate_limit" in err_msg.lower():
                time.sleep(2.0 * attempt)
            elif attempt < max_retries:
                time.sleep(1.0)
            else:
                return {
                    "stance": "INSUFFICIENT_EVIDENCE",
                    "conflicting_attribute": None,
                    "reasoning": f"Stance could not be verified due to API connection limit: {err_msg[:60]}",
                    "confidence": 0.50,
                }


def build_cluster_stance_edges(
    clusters: Dict[int, Dict[str, Any]],
    client: Optional[Groq] = None
) -> Dict[str, Any]:
    """
    Evaluates pairwise stances between all cross-source claims within each cluster.
    """
    if client is None:
        api_key = get_groq_api_key()
        client = Groq(api_key=api_key) if api_key else None

    total_pairs = 0
    disagree_pairs = 0
    agree_pairs = 0

    results = {}

    for cid, cdata in clusters.items():
        claims = cdata["claims"]
        if len(claims) < 2:
            results[str(cid)] = {
                "cluster_id": cid,
                "topic": cdata.get("topic", "general"),
                "claims": claims,
                "edges": []
            }
            continue

        edges = []
        # Generate cross-source pairs only (different sources)
        cross_pairs = [
            (a, b) for a, b in itertools.combinations(claims, 2)
            if a.get("source_name") != b.get("source_name")
        ]

        seen_pairs = set()
        for a, b in cross_pairs:
            pair_key = tuple(sorted([a.get("claim_id", a["claim"]), b.get("claim_id", b["claim"])]))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            total_pairs += 1
            stance_res = classify_stance_pair(a, b, client=client)

            stance = stance_res["stance"]
            if stance == "DISAGREE":
                disagree_pairs += 1
            elif stance == "AGREE":
                agree_pairs += 1

            edges.append({
                "pair_id": f"pair_{a.get('claim_id')}__{b.get('claim_id')}",
                "claim_a_id": a.get("claim_id"),
                "claim_a": a.get("claim"),
                "source_a": a.get("source_name") or a.get("source_file"),
                "date_a": a.get("date"),
                "val_a": a.get("numerical_value"),
                "unit_a": a.get("numerical_unit"),
                "claim_b_id": b.get("claim_id"),
                "claim_b": b.get("claim"),
                "source_b": b.get("source_name") or b.get("source_file"),
                "date_b": b.get("date"),
                "val_b": b.get("numerical_value"),
                "unit_b": b.get("numerical_unit"),
                "stance": stance,
                "conflicting_attribute": stance_res.get("conflicting_attribute"),
                "reasoning": stance_res.get("reasoning", ""),
                "confidence": stance_res.get("confidence", 0.85),
            })

            # Small pause if LLM was called to maintain rate limit headroom
            time.sleep(0.1)

        results[str(cid)] = {
            "cluster_id": cid,
            "topic": cdata.get("topic", "general"),
            "claims": claims,
            "edges": edges
        }

    print(f"\nStance Evaluation Complete: {total_pairs} pairs evaluated ({disagree_pairs} DISAGREE, {agree_pairs} AGREE).")
    return results


def run_stance_pipeline(
    claims_path: str = CLAIMS_FILE,
    output_path: str = CLUSTERS_FILE,
    distance_threshold: Optional[float] = None
) -> Tuple[int, Dict[str, Any], float]:
    """
    Loads claims, performs clustering, evaluates pairwise stances, and saves output.
    """
    if not os.path.exists(claims_path):
        raise FileNotFoundError(f"{claims_path} not found. Run Step 1 (Claim Extraction) first.")

    with open(claims_path, "r", encoding="utf-8") as f:
        claims = json.load(f)

    print(f"Loaded {len(claims)} atomic claims from {claims_path}.")
    print("Executing Semantic Clustering...")
    clusters, used_threshold = cluster_claims(claims, distance_threshold=distance_threshold)
    print(f"Formed {len(clusters)} fact clusters (Distance Threshold: {used_threshold:.2f}).")

    print("Evaluating Cross-Source Stances (Neuro-Symbolic + LLM CoT)...")
    results = build_cluster_stance_edges(clusters)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[DONE] Successfully saved clustered stances to {output_path}")
    return len(clusters), results, used_threshold


if __name__ == "__main__":
    run_stance_pipeline()
