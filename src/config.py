"""
Central Configuration Module for Contradiction-Aware News Analysis
------------------------------------------------------------------
Contains all paths, model identifiers, API key loaders, and pipeline hyperparameters.
"""

import os
import sys
from dotenv import load_dotenv

# Ensure the root directory of the project is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Automatically load environment variables from .env in project root
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")
if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)
else:
    load_dotenv()

# ==============================================================================
# DIRECTORY & FILE PATHS
# ==============================================================================
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_SOURCES_DIR = os.path.join(DATA_DIR, "raw_sources")
CUSTOM_SOURCES_DIR = os.path.join(DATA_DIR, "custom_sources")
BENCHMARK_DIR = os.path.join(DATA_DIR, "evaluation_benchmark")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
CLAIMS_FILE = os.path.join(OUTPUT_DIR, "claims.json")
CLUSTERS_FILE = os.path.join(OUTPUT_DIR, "clusters_with_stance.json")
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "final_summary.md")

# Ensure required runtime directories exist
for directory in [DATA_DIR, RAW_SOURCES_DIR, CUSTOM_SOURCES_DIR, BENCHMARK_DIR, OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)

# ==============================================================================
# MODEL IDENTIFIERS
# ==============================================================================
# Primary Groq LLM model verified in environment
GROQ_MODEL = "qwen/qwen3.8-27b"

# Sentence embedding model for clustering
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ==============================================================================
# PIPELINE HYPERPARAMETERS
# ==============================================================================
# Max tokens for Groq chat completions
EXTRACTION_MAX_TOKENS = 1500  # Upgraded from 350 to prevent truncated JSON
STANCE_MAX_TOKENS = 350       # Upgraded from 90 to prevent truncated CoT reasoning

# Clustering parameters
DEFAULT_DISTANCE_THRESHOLD = 0.55
CANDIDATE_THRESHOLDS = (0.35, 0.45, 0.50, 0.55, 0.60, 0.70)

# 5-Class Stance Schema
STANCE_LABELS = [
    "AGREE",
    "DISAGREE",
    "DISCUSS",
    "UNRELATED",
    "INSUFFICIENT_EVIDENCE"
]

# ==============================================================================
# GROQ API CLIENT HELPER
# ==============================================================================
def get_groq_api_key() -> str:
    """
    Safely retrieves the Groq API key from environment, .env file, or Streamlit secrets.
    """
    key = os.getenv("GROQ_API_KEY", "").strip()
    if key:
        return key

    # Check Streamlit Cloud / local secrets if running inside streamlit
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
            key = str(st.secrets["GROQ_API_KEY"]).strip()
            os.environ["GROQ_API_KEY"] = key
            return key
    except Exception:
        pass

    return ""
