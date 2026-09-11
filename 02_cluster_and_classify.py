"""
Step 2: Claim Clustering + Stance Classification
--------------------------------------------------
Reads output/claims.json (from step 1), clusters claims that talk about the
same underlying fact (e.g. "death toll", "people affected"), then uses
Groq LLM to classify each cross-source pair within a cluster as Agree / Disagree / Discuss / Unrelated
(same task formulation as the FNC-1 Fake News Challenge stance task).

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
    raise ValueError("GROQ_API_KEY is not set in .env")

client = Groq(api_key=api_key)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

MODEL_NAME = "qwen/qwen3.8-27b"

STANCE_PROMPT = """You are a stance classifier following the FNC-1 (Fake News Challenge) task definition.
Given two factual claims about the 2026 Assam Floods, classify the relationship between Claim B and Claim A as exactly one of:

- "agree": Claim B confirms, supports, or is substantively consistent with Claim A (same facts, same impacts, or corroborating status).
- "disagree": Claim B contradicts Claim A (conflicting casualty figures for the same event, conflicting people affected counts e.g. 1.78L vs 7.2L, or contradictory official totals e.g. 47 vs 82/100).
- "discuss": Claim B is about the same topic but neither confirms nor contradicts directly. CRITICAL: If Claim A and Claim B report death tolls from DIFFERENT DATES showing numbers increasing chronologically over time (e.g. 66 on 26 July vs 100 on 10 August), this is temporal progression, NOT a contradiction. Classify as "discuss".
- "unrelated": Claim B is about a completely different fact or detail than Claim A.

Claim A (from {source_a}, date: {date_a}): "{claim_a}"
Claim B (from {source_b}, date: {date_b}): "{claim_b}"

Respond with ONLY one word: agree, disagree, discuss, or unrelated.
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


def cluster_claims(claims: list, distance_threshold: float = 0.55):
    """Clusters claims using cosine distance on SentenceTransformer embeddings."""
    cleaned_texts = [clean_for_embedding(c["claim"]) for c in claims]
    embeddings = embedder.encode(cleaned_texts, show_progress_bar=False)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(embeddings)

    for c, label in zip(claims, labels):
        c["cluster_id"] = int(label)

    return claims


def classify_stance(claim_a: dict, claim_b: dict, max_retries: int = 5) -> str:
    """Classifies the stance between two claims using Groq with backoff."""
    prompt = STANCE_PROMPT.format(
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
                max_tokens=10,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a stance classifier. Output only one single word: agree, disagree, discuss, or unrelated."
                    },
                    {"role": "user", "content": prompt}
                ],
            )
            raw = response.choices[0].message.content.strip().lower()
            clean_word = raw.split()[0].strip(".,!?:;\"'()`*") if raw.split() else ""
            if clean_word in ["agree", "disagree", "discuss", "unrelated"]:
                return clean_word
            
            for valid in ["agree", "disagree", "discuss", "unrelated"]:
                if valid in raw:
                    return valid

            return "discuss"
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
                return "discuss"


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
        for a, b in itertools.combinations(members, 2):
            # Skip pairs from the same source file (not a cross-source contradiction)
            if a["source_file"] == b["source_file"]:
                continue

            total_pairs += 1
            stance = classify_stance(a, b)
            time.sleep(0.35)  # slight pace to respect Groq rate limits
            if stance == "disagree":
                disagree_pairs += 1

            edges.append({
                "claim_a": a["claim"],
                "source_a": a["source_file"],
                "date_a": a.get("date"),
                "claim_b": b["claim"],
                "source_b": b["source_file"],
                "date_b": b.get("date"),
                "stance": stance,
            })

        results[cluster_id] = {"claims": members, "edges": edges}

    print(f"Evaluated {total_pairs} cross-source pairs ({disagree_pairs} disagree, {total_pairs - disagree_pairs} other).")
    return results


def run_pipeline_step2(distance_threshold: float = 0.55):
    input_path = "output/claims.json"
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path} not found. Run Step 1 first.")

    with open(input_path, "r", encoding="utf-8") as f:
        claims = json.load(f)

    print(f"Loaded {len(claims)} claims. Clustering with distance_threshold={distance_threshold}...")
    claims = cluster_claims(claims, distance_threshold=distance_threshold)
    n_clusters = len(set(c["cluster_id"] for c in claims))
    print(f"Formed {n_clusters} clusters.")

    print("Classifying stance within cross-source pairs in each cluster...")
    results = add_stance_edges(claims)

    os.makedirs("output", exist_ok=True)
    output_path = "output/clusters_with_stance.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Done. Saved to {output_path}")
    return n_clusters, results


def main():
    import sys
    threshold = 0.55
    if len(sys.argv) > 1:
        try:
            threshold = float(sys.argv[1])
        except ValueError:
            pass
    run_pipeline_step2(distance_threshold=threshold)


if __name__ == "__main__":
    main()
