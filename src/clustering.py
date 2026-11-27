"""
Semantic Claim Clustering Module
---------------------------------
Converts extracted factual claims into dense semantic embeddings using
Sentence-Transformers (all-MiniLM-L6-v2) and groups them using
Agglomerative Hierarchical Clustering with cosine distance.

Features:
- Domain-agnostic text preparation (no hardcoded event strings)
- Adaptive silhouette score threshold optimization
- Safe fallback to configurable default threshold
"""

import re
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from src.config import (
    EMBEDDING_MODEL,
    DEFAULT_DISTANCE_THRESHOLD,
    CANDIDATE_THRESHOLDS,
)

# Global embedder instance (lazy loaded on first use)
_EMBEDDER: Optional[SentenceTransformer] = None


def get_embedder() -> SentenceTransformer:
    """Returns the cached SentenceTransformer singleton instance."""
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer(EMBEDDING_MODEL)
    return _EMBEDDER


def clean_text_for_embedding(text: str) -> str:
    """
    Cleans journalistic boilerplates, metadata lead-ins, and redundant filler words
    in a completely domain-agnostic manner.
    """
    cleaned = text.strip()
    # Strip common journalistic attribution prefixes
    cleaned = re.sub(
        r"^(according to [^,]+,|officials stated that|reports indicate that|sources confirmed that|as per reports,?)\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    # Strip temporal anchors at the end/beginning that might skew topical semantic similarity
    cleaned = re.sub(r"\b(as of|on)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|yesterday|today)\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if cleaned else text


def prepare_text_for_embedding(claim_dict: Dict[str, Any]) -> str:
    """
    Constructs a rich hybrid representation incorporating topic category,
    semantic subject, and normalized claim text.
    """
    raw_text = claim_dict.get("normalized_claim") or claim_dict.get("claim", "")
    clean_text = clean_text_for_embedding(raw_text)

    topic = str(claim_dict.get("category") or claim_dict.get("topic") or "").strip().lower()
    unit = str(claim_dict.get("numerical_unit") or "").strip().lower()

    structured = claim_dict.get("structured_attributes")
    subject = structured.get("subject") if isinstance(structured, dict) else None

    components = []
    if topic and topic != "general":
        components.append(f"[{topic}]")
    if unit:
        components.append(f"({unit})")
    if subject and subject.lower() not in clean_text.lower():
        components.append(f"{subject}:")

    components.append(clean_text)
    return " ".join(components).strip()


def find_optimal_threshold(
    embeddings,
    candidate_thresholds: Tuple[float, ...] = CANDIDATE_THRESHOLDS,
    default_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
) -> float:
    """
    Dynamically searches for the clustering distance threshold that maximizes
    the silhouette score. Gracefully falls back to default_threshold if the
    corpus is too small or scores are ambiguous.
    """
    n_samples = len(embeddings)
    if n_samples < 4:
        return default_threshold

    best_threshold = default_threshold
    best_score = -1.0

    for threshold in candidate_thresholds:
        try:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=threshold,
                metric="cosine",
                linkage="average",
            )
            labels = clustering.fit_predict(embeddings)
            n_clusters = len(set(labels))
            if 1 < n_clusters < n_samples:
                score = silhouette_score(embeddings, labels, metric="cosine")
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
        except Exception:
            continue

    if best_score > 0.05:
        print(f"Optimal cluster distance auto-selected: {best_threshold:.2f} (Silhouette Score: {best_score:.3f})")
        return best_threshold

    return default_threshold


def cluster_claims(
    claims: List[Dict[str, Any]],
    distance_threshold: Optional[float] = None,
    embedder: Optional[SentenceTransformer] = None,
) -> Tuple[Dict[int, Dict[str, Any]], float]:
    """
    Clusters a list of claims using cosine distance on SentenceTransformer embeddings.
    Returns:
        (clusters_dict, used_distance_threshold)
    """
    if not claims:
        return {}, distance_threshold or DEFAULT_DISTANCE_THRESHOLD

    if len(claims) == 1:
        claims[0]["cluster_id"] = 0
        single_cluster = {
            0: {
                "cluster_id": 0,
                "topic": claims[0].get("category") or claims[0].get("topic", "general"),
                "claims": claims
            }
        }
        return single_cluster, distance_threshold or DEFAULT_DISTANCE_THRESHOLD

    if embedder is None:
        embedder = get_embedder()

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

    clusters: Dict[int, Dict[str, Any]] = {}
    for c, label in zip(claims, labels):
        cid = int(label)
        c["cluster_id"] = cid
        if cid not in clusters:
            clusters[cid] = {
                "cluster_id": cid,
                "topic": c.get("category") or c.get("topic", "general"),
                "claims": []
            }
        clusters[cid]["claims"].append(c)

    return clusters, distance_threshold


def test_clustering_on_sample_pairs() -> bool:
    """
    Validation unit test for semantic claim clustering.
    Confirms that:
    1. Semantically identical claims cluster together.
    2. Factually distinct claims do not cluster together.
    """
    sample_claims = [
        {
            "claim": "Five people were injured.",
            "category": "casualties",
            "numerical_unit": "injured",
            "structured_attributes": {"subject": "people"}
        },
        {
            "claim": "5 persons sustained injuries.",
            "category": "casualties",
            "numerical_unit": "injured",
            "structured_attributes": {"subject": "persons"}
        },
        {
            "claim": "Rescue teams reached the location.",
            "category": "rescue_relief",
            "numerical_unit": None,
            "structured_attributes": {"subject": "rescue teams"}
        }
    ]

    clusters, _ = cluster_claims(sample_claims, distance_threshold=0.55)

    c0_id = sample_claims[0]["cluster_id"]
    c1_id = sample_claims[1]["cluster_id"]
    c2_id = sample_claims[2]["cluster_id"]

    same_fact_passed = (c0_id == c1_id)
    different_fact_passed = (c0_id != c2_id)

    print(f"Test 1 (Same fact clustered): {'PASS' if same_fact_passed else 'FAIL'}")
    print(f"Test 2 (Different fact separated): {'PASS' if different_fact_passed else 'FAIL'}")

    return same_fact_passed and different_fact_passed


if __name__ == "__main__":
    print("Testing Semantic Clustering on Sample Cases...")
    test_clustering_on_sample_pairs()
