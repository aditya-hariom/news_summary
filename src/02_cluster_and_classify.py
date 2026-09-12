"""
Step 2: Adaptive Claim Clustering + Neuro-Symbolic & CoT Stance Classification
-------------------------------------------------------------------------------
1. Converts claims into dense embeddings using SentenceTransformers.
2. Dynamically searches for the optimal distance threshold using Silhouette Scoring
   (or uses user-supplied threshold).
3. Applies Neuro-Symbolic rules for temporal & numerical validation.
4. Uses Groq LLM with Chain-of-Thought (CoT) prompting to classify cross-source pairs
   into: agree, disagree, discuss, or unrelated with explicit reasoning & confidence scores.

Run: python src/02_cluster_and_classify.py
Output: output/clusters_with_stance.json
"""

import os
import re
import json
import itertools
import time
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from groq import Groq

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
        api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
            api_key = str(st.secrets["GROQ_API_KEY"]).strip()
            os.environ["GROQ_API_KEY"] = api_key
    except Exception:
        pass

if not api_key:
    raise ValueError("GROQ_API_KEY is not set in .env or Streamlit Cloud Secrets.")

client = Groq(api_key=api_key)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

MODEL_NAME = "qwen/qwen3.8-27b"

COT_STANCE_PROMPT = """You are a stance classifier following the FNC-1 (Fake News Challenge) task definition.
Analyze the factual relationship between Claim B and Claim A regarding the same incident or situation.

Categories:
- "agree": Claim B confirms, supports, or is substantively consistent with Claim A (same facts, corroborated counts, or same status).
- "disagree": Claim B directly contradicts Claim A (incompatible numerical figures, opposite outcomes, or conflicting official accounts for the same event window).
- "discuss": Claim B discusses the same topic but neither strictly confirms nor contradicts (e.g. statistics from DIFFERENT DATES showing chronological progression, or supplementary details).
- "unrelated": Claim B is about a completely different detail, metric, or subject.

Claim A (from {source_a}, date: {date_a}): "{claim_a}"
Claim B (from {source_b}, date: {date_b}): "{claim_b}"

Rules for Output:
1. Provide concise step-by-step reasoning (1-2 sentences) comparing facts, figures, and dates.
2. Choose exactly one stance: agree, disagree, discuss, or unrelated.
3. Assign a confidence score between 0.0 and 1.0.
4. Output valid JSON only.

JSON Format:
{{
  "reasoning": "1-2 sentence comparison of figures and facts",
  "stance": "agree | disagree | discuss | unrelated",
  "confidence": 0.95
}}
"""


def clean_for_embedding(text: str) -> str:
    """Removes repetitive event-level boilerplate to let topical semantics dominate embeddings."""
    cleaned = re.sub(
        r"\b(in|by|across|during|for)?\s*(the\s*)?(2026\s*)?Assam\s*(floods?|state)?\b",
        "",
        text,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if cleaned else text


def prepare_text_for_embedding(c: dict) -> str:
    """Creates a hybrid text representation incorporating topic and unit for cleaner clustering."""
    clean_claim = clean_for_embedding(c["claim"])
    topic = c.get("topic")
    unit = c.get("numerical_unit")
    prefix = ""
    if topic and topic != "general":
        prefix += f"[{topic}] "
    if unit:
        prefix += f"({unit}) "
    return f"{prefix}{clean_claim}".strip()


def find_optimal_threshold(embeddings, candidate_thresholds=(0.35, 0.45, 0.50, 0.55, 0.60, 0.70), default_threshold=0.55) -> float:
    """
    Dynamically searches for the clustering threshold that maximizes the silhouette score.
    Falls back gracefully to default_threshold if dataset is small or scores are ambiguous.
    """
    n_samples = len(embeddings)
    if n_samples < 4:
        return default_threshold

    best_threshold = default_threshold
    best_score = -1.0

    for t in candidate_thresholds:
        try:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=t,
                metric="cosine",
                linkage="average",
            )
            labels = clustering.fit_predict(embeddings)
            n_clusters = len(set(labels))
            if 1 < n_clusters < n_samples:
                score = silhouette_score(embeddings, labels, metric="cosine")
                if score > best_score:
                    best_score = score
                    best_threshold = t
        except Exception:
            continue

    if best_score > 0.05:
        print(f"Optimal cluster distance threshold auto-selected: {best_threshold:.2f} (Silhouette Score: {best_score:.3f})")
        return best_threshold
    return default_threshold


def cluster_claims(claims: list, distance_threshold: float = None):
    """Clusters claims using cosine distance on SentenceTransformer embeddings."""
    hybrid_texts = [prepare_text_for_embedding(c) for c in claims]
    embeddings = embedder.encode(hybrid_texts, show_progress_bar=False)

    if distance_threshold is None:
        distance_threshold = find_optimal_threshold(embeddings)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(embeddings)

    for c, label in zip(claims, labels):
        c["cluster_id"] = int(label)

    return claims, distance_threshold


def deterministic_stance_check(claim_a: dict, claim_b: dict):
    """
    Applies neuro-symbolic rules on explicit numbers, units, topics, and texts.
    Returns a dict with stance, reasoning, confidence if a clear deterministic pattern exists.
    """
    # Rule 0: Exact or normalized text match
    norm_a = re.sub(r"[^\w\s]", "", claim_a["claim"]).strip().lower()
    norm_b = re.sub(r"[^\w\s]", "", claim_b["claim"]).strip().lower()
    if norm_a == norm_b:
        return {
            "stance": "agree",
            "reasoning": "Identical factual claim corroborated across multiple independent outlets.",
            "confidence": 0.99,
        }

    # Rule 1: Divergent topic categories
    top_a = claim_a.get("topic")
    top_b = claim_b.get("topic")
    if top_a and top_b and top_a != "general" and top_b != "general" and top_a != top_b:
        return {
            "stance": "unrelated",
            "reasoning": f"Different factual topic domains ({top_a} vs {top_b}).",
            "confidence": 0.95,
        }

    val_a = claim_a.get("numerical_value")
    val_b = claim_b.get("numerical_value")
    date_a = str(claim_a.get("date") or "").strip().lower()
    date_b = str(claim_b.get("date") or "").strip().lower()
    unit_a = str(claim_a.get("numerical_unit") or "").strip().lower()
    unit_b = str(claim_b.get("numerical_unit") or "").strip().lower()

    if val_a is not None and val_b is not None:
        try:
            num_a = float(val_a)
            num_b = float(val_b)
            if num_a > 0 and num_b > 0:
                ratio = max(num_a, num_b) / min(num_a, num_b)

                # Rule 2: Exactly identical numbers on same metric
                if ratio == 1.0 and (not unit_a or not unit_b or unit_a == unit_b):
                    return {
                        "stance": "agree",
                        "reasoning": f"Exact matching statistics ({num_a:g}) confirmed by both outlets.",
                        "confidence": 0.98,
                    }

                # Rule 3: High numerical divergence for same timeframe (> 25% disparity)
                if (date_a == date_b or not date_a or not date_b) and ratio >= 1.25:
                    if not unit_a or not unit_b or unit_a == unit_b:
                        return {
                            "stance": "disagree",
                            "reasoning": f"Numerical contradiction: {claim_a.get('source_file')} reports {num_a:g} vs {claim_b.get('source_file')} reports {num_b:g} ({ratio:.1f}x disparity) for the same event window.",
                            "confidence": 0.95,
                        }

                # Rule 4: Chronological progression across distinct dates
                if date_a and date_b and date_a != date_b and ratio >= 1.05:
                    return {
                        "stance": "discuss",
                        "reasoning": f"Temporal progression: {claim_a.get('source_file')} reported {num_a:g} ({date_a}) and {claim_b.get('source_file')} reported {num_b:g} ({date_b}). Figures naturally progress across dates.",
                        "confidence": 0.90,
                    }
        except (ValueError, TypeError):
            pass

    return None


def classify_stance(claim_a: dict, claim_b: dict, max_retries: int = 4) -> dict:
    """Classifies the stance between two claims using deterministic rules or Groq LLM with CoT."""
    # 1. Deterministic Rule Check
    rule_res = deterministic_stance_check(claim_a, claim_b)
    if rule_res is not None:
        return rule_res

    # 2. LLM Chain-of-Thought
    prompt = COT_STANCE_PROMPT.format(
        claim_a=claim_a["claim"],
        source_a=claim_a.get("source_file", "unknown"),
        date_a=claim_a.get("date", "not specified"),
        claim_b=claim_b["claim"],
        source_b=claim_b.get("source_file", "unknown"),
        date_b=claim_b.get("date", "not specified"),
    )

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=0.0,
                max_tokens=90,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a stance classifier. Output valid JSON only with 'reasoning', 'stance', and 'confidence'."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            raw = response.choices[0].message.content.strip()
            data = json.loads(raw)
            stance = str(data.get("stance", "discuss")).strip().lower()
            if stance not in ["agree", "disagree", "discuss", "unrelated"]:
                for valid in ["agree", "disagree", "discuss", "unrelated"]:
                    if valid in stance:
                        stance = valid
                        break
                else:
                    stance = "discuss"

            return {
                "stance": stance,
                "reasoning": data.get("reasoning", "Cross-source comparison evaluated by model."),
                "confidence": float(data.get("confidence", 0.85)),
            }
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "rate_limit" in err_msg.lower():
                wait_time = 2.0 * attempt
                print(f"  [Rate limit] Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            elif attempt < max_retries:
                time.sleep(1.0)
            else:
                print(f"[WARN] Error classifying stance: {e}. Falling back to 'discuss'.")
                return {
                    "stance": "discuss",
                    "reasoning": "Fallback classification due to API interruption.",
                    "confidence": 0.50,
                }


def add_stance_edges(claims: list) -> dict:
    clusters = {}
    for c in claims:
        clusters.setdefault(c["cluster_id"], []).append(c)

    results = {}
    total_pairs = 0
    disagree_pairs = 0

    for cluster_id, members in clusters.items():
        if len(members) < 2:
            results[cluster_id] = {"claims": members, "edges": []}
            continue

        edges = []
        possible_pairs = [
            (a, b) for a, b in itertools.combinations(members, 2)
            if a["source_file"] != b["source_file"]
        ]

        # Limit to max 12 most representative pairs per cluster to prevent rate limit spikes
        if len(possible_pairs) > 12:
            possible_pairs = possible_pairs[:12]

        for a, b in possible_pairs:
            total_pairs += 1
            stance_res = classify_stance(a, b)
            # Brief pause only if called LLM
            if "disparity" not in stance_res.get("reasoning", "") and "progression" not in stance_res.get("reasoning", ""):
                time.sleep(0.4)

            if stance_res["stance"] == "disagree":
                disagree_pairs += 1

            edges.append({
                "claim_a": a["claim"],
                "source_a": a["source_file"],
                "date_a": a.get("date"),
                "claim_b": b["claim"],
                "source_b": b["source_file"],
                "date_b": b.get("date"),
                "stance": stance_res["stance"],
                "reasoning": stance_res.get("reasoning", ""),
                "confidence": stance_res.get("confidence", 0.85),
            })

        results[cluster_id] = {"claims": members, "edges": edges}

    print(f"Evaluated {total_pairs} cross-source pairs ({disagree_pairs} disagree, {total_pairs - disagree_pairs} other).")
    return results


def run_pipeline_step2(distance_threshold: float = None):
    input_path = "output/claims.json"
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path} not found. Run Step 1 first.")

    with open(input_path, "r", encoding="utf-8") as f:
        claims = json.load(f)

    print(f"Loaded {len(claims)} claims. Clustering...")
    claims, used_threshold = cluster_claims(claims, distance_threshold=distance_threshold)
    n_clusters = len(set(c["cluster_id"] for c in claims))
    print(f"Formed {n_clusters} clusters (Distance Threshold: {used_threshold:.2f}).")

    print("Classifying stance with Neuro-Symbolic rules & Chain-of-Thought LLM...")
    results = add_stance_edges(claims)

    os.makedirs("output", exist_ok=True)
    output_path = "output/clusters_with_stance.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Done. Saved to {output_path}")
    return n_clusters, results, used_threshold


def main():
    import sys
    threshold = None
    if len(sys.argv) > 1:
        try:
            threshold = float(sys.argv[1])
        except ValueError:
            pass
    run_pipeline_step2(distance_threshold=threshold)


if __name__ == "__main__":
    main()
