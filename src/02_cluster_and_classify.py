"""
Step 2: Clustering and Stance Classification Runner (Backward Compatible Shim)
------------------------------------------------------------------------------
Delegates execution to the modular src.clustering and src.stance_classifier engines.

Run: python src/02_cluster_and_classify.py
Output: output/clusters_with_stance.json
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.stance_classifier import (
    run_stance_pipeline,
    classify_stance_pair,
    deterministic_stance_check,
    build_cluster_stance_edges,
)
from src.clustering import cluster_claims


def run_pipeline_step2(distance_threshold=None):
    """Entry point called by Streamlit UI and pipeline runners."""
    return run_stance_pipeline(distance_threshold=distance_threshold)


def main():
    threshold = None
    if len(sys.argv) > 1:
        try:
            threshold = float(sys.argv[1])
        except ValueError:
            pass
    run_pipeline_step2(distance_threshold=threshold)


if __name__ == "__main__":
    main()
