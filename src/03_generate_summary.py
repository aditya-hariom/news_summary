"""
Step 3: Contradiction-Aware Summary Runner (Backward Compatible Shim)
---------------------------------------------------------------------
Delegates synthesis to the modular src.summarizer engine.

Run: python src/03_generate_summary.py
Output: output/final_summary.md
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.summarizer import (
    run_summary_pipeline,
    categorize_clusters,
    render_markdown_report,
)


def run_pipeline_step3():
    """Entry point called by Streamlit UI and CLI."""
    return run_summary_pipeline()


def main():
    run_pipeline_step3()


if __name__ == "__main__":
    main()
