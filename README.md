# Contradiction-Aware Multi-Document News Analysis Using LLMs

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://newssummary-zzev4ju2gta656xhj4vtwa.streamlit.app/)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)

An AI-powered NLP pipeline and interactive Streamlit dashboard that analyzes multiple conflicting news articles covering breaking events, extracts atomic factual claims, clusters them by topic, detects agreements and contradictions using a neuro-symbolic hybrid stance classifier, and generates transparent, contradiction-aware summaries with full source attribution.

🌐 **Live Demo:** https://newssummary-zzev4ju2gta656xhj4vtwa.streamlit.app/

---

## 📌 Problem Statement & Novelty

When breaking news unfolds, different media outlets frequently publish conflicting casualty figures, dates, and affected-population statistics.

- **Standard AI Summarizers (ChatGPT / Gemini / T5):** Blend conflicting reports into a vague consensus (*"Between 47 and 100 people died..."*), hiding reporting discrepancies and creating hallucinated compromises.
- **Our System:** Isolates individual factual claims, clusters them by semantic topic, separates timeline progressions (`DISCUSS`) from true contradictions (`DISAGREE`), and displays conflicting statements **side-by-side** with exact outlet names, dates, and reasoning — never picking a winner by majority vote.

---

## 🏗️ System Architecture

```text
Multiple News Outlets (Pasted Text / .txt Uploads / Built-in Corpus)
                    │
                    ▼
    [1. Atomic Claim Extraction & Normalization]
      Groq LLM (qwen/qwen3.8-27b) + Unit/Number Normalizer
                    │
                    ▼
          [output/claims.json]
                    │
                    ▼
    [2. Semantic Topic Clustering]
      Sentence-Transformers (all-MiniLM-L6-v2)
      + Agglomerative Clustering (Cosine Distance)
      + Adaptive Silhouette Score Search
                    │
                    ▼
    [3. Neuro-Symbolic Stance & Contradiction Engine]
      Deterministic Rule Engine (Units, Numeric Disparities, Temporal Anchors)
      + Groq LLM Chain-of-Thought (5-Class FNC-1 Schema)
                    │
                    ▼
      [output/clusters_with_stance.json]
                    │
                    ▼
    [4. Contradiction-Aware Summary Synthesis]
      Categorization: Corroborated, Disputed, Developing, Unverified
                    │
                    ▼
         [output/final_summary.md]
                    │
                    ▼
   [5. Guided Two-View Streamlit Dashboard (app.py)]
```

---

## 🖥️ Dashboard Highlights (`app.py`)

The Streamlit interface is organized into a clean two-view layout:

1. **🔍 Compare & Results**
   - **Step 1 — Collapsible Input Panel:** Paste two articles side-by-side, upload 2+ `.txt` files, or run the built-in 4-source 2026 Assam Floods corpus (with 1-click quick demo presets in the sidebar).
   - **Step 2 — Comparison Results:** Live 3-stage pipeline progress (`st.status`), at-a-glance verdict banner and metrics, and four focused result tabs:
     - **🔴 Conflicts (`DISAGREE`):** Outlet A vs Outlet B side-by-side comparison cards with conflicting attribute badges and reasoning.
     - **🟢 Agreements (`AGREE`):** Multi-source corroborated facts with source chips.
     - **🟡 Updates & Single-Source (`DISCUSS` & Unverified):** Chronological/context revisions separated cleanly from single-source claims.
     - **📄 Full Report:** Rendered Markdown summary with one-click `.md` download.
2. **📚 Claims Explorer & Guide**
   - Search and filter individual extracted claims by keyword, outlet, or topic.
   - Architecture overview, 5-class stance label guide, and raw source article viewer.

---

## 🚀 Quickstart

### 1. Prerequisites
- Python 3.11 (see [`runtime.txt`](runtime.txt))
- Groq API Key from [console.groq.com](https://console.groq.com)

### 2. Clone & Install Dependencies
```bash
git clone https://github.com/aditya-hariom/news_summary.git
cd news_summary

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure API Key
Copy [`.env.example`](.env.example) to `.env` in the project root:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
```
*(For Streamlit Cloud deployment, configure `GROQ_API_KEY` in Secrets using [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example) as a template — or enter it directly in the dashboard sidebar.)*

### 4. Launch the Dashboard
```bash
streamlit run app.py
```

---

## 💻 Running the Pipeline & Tests via CLI

Run each pipeline stage independently from the project root:

```bash
# Step 1: Extract atomic factual claims -> output/claims.json
python src/extraction.py

# Step 2: Cluster claims & classify cross-source stances -> output/clusters_with_stance.json
python src/stance_classifier.py

# Step 3: Synthesize contradiction-aware Markdown report -> output/final_summary.md
python src/summarizer.py

# Step 4: Run benchmark evaluation -> output/evaluation_report.json
python src/evaluation.py

# Step 5: Run edge-case & runtime verification suite
python tests/test_edge_cases.py
```

---

## 📊 Evaluation & Benchmark Results

### Built-in 4-Source 2026 Assam Floods Corpus (`output/final_summary.md`)
- **Corroborated Topics:** 1 | **Active Disputes:** 1 | **Developing Updates:** 2 | **Single-Source Claims:** 6

### Stance Classification Benchmark (`output/evaluation_report.json`)
Evaluated against 20 manually labeled claim pairs in [`data/evaluation_benchmark/benchmark_pairs.json`](data/evaluation_benchmark/benchmark_pairs.json):

| Stance Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **AGREE** | 1.00 | 1.00 | **1.00** | 5 |
| **DISAGREE** | 0.83 | 0.83 | **0.83** | 6 |
| **DISCUSS** | 0.50 | 1.00 | **0.67** | 4 |
| **UNRELATED** | 1.00 | 0.33 | **0.50** | 3 |
| **INSUFFICIENT_EVIDENCE** | 0.00 | 0.00 | **0.00** | 2 |
| **Overall Accuracy** | — | — | **75.0%** | 20 |
| **Macro Average** | 0.667 | 0.633 | **0.600** | — |

---

## 📁 Repository Structure

```text
news_summary/
├── app.py                                # Guided two-view Streamlit dashboard
├── requirements.txt                      # Python dependencies
├── runtime.txt                           # Python runtime version (python-3.11)
├── .env.example                          # Environment variable template
├── .gitignore                            # Ignore rules for venv, secrets, temp uploads
├── README.md                             # Project documentation
│
├── .streamlit/
│   ├── config.toml                       # Streamlit theme & server config
│   └── secrets.toml.example              # Streamlit Cloud secrets template
│
├── data/
│   ├── raw_sources/                      # 2026 Assam floods corpus (4 outlets + map)
│   │   ├── README_contradiction_map.txt
│   │   ├── source1_wikipedia_compiled.txt
│   │   ├── source2_npr.txt
│   │   ├── source3_theunitedindian.txt
│   │   └── source4_mapsofindia.txt
│   ├── custom_sources/                   # Ephemeral directory for pasted/uploaded inputs
│   └── evaluation_benchmark/
│       └── benchmark_pairs.json          # 20 labeled stance evaluation pairs
│
├── output/
│   ├── claims.json                       # Extracted atomic claims
│   ├── clusters_with_stance.json         # Semantic clusters & stance edges
│   ├── final_summary.md                  # Generated contradiction-aware report
│   └── evaluation_report.json            # Benchmark metrics & confusion matrix
│
├── src/
│   ├── __init__.py
│   ├── config.py                         # Paths, models, hyperparameters, key loader
│   ├── extraction.py                     # LLM claim extraction & number normalization
│   ├── clustering.py                     # Sentence-transformer agglomerative clustering
│   ├── stance_classifier.py              # Neuro-symbolic stance & contradiction engine
│   ├── summarizer.py                     # Categorization & Markdown report synthesis
│   └── evaluation.py                     # Ground-truth benchmark evaluation script
│
└── tests/
    └── test_edge_cases.py                # 10-case edge & runtime verification suite
```

---

## 👤 Author
Developed by **Aditya** (2026).
