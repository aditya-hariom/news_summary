"""
Step 1: Claim Extraction Runner (Backward Compatible Shim)
-----------------------------------------------------------
Delegates execution to the modular src.extraction engine.

Run: python src/01_extract_claims.py
Output: output/claims.json
"""

import sys
import os

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.extraction import (
    extract_claims_from_file,
    extract_claims_from_text,
    parse_claims_json,
    run_extraction_pipeline,
)


def main(input_files=None):
    """Entry point compatible with both CLI and Streamlit importlib."""
    return run_extraction_pipeline(input_files=input_files)


if __name__ == "__main__":
    main()
