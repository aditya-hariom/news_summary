# Contradiction-Aware Multi-Document News Analysis Using LLMs

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://newssummary-zzev4ju2gta656xhj4vtwa.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An AI-powered NLP pipeline and interactive Streamlit dashboard that analyzes multiple conflicting news articles covering breaking events, extracts atomic factual claims, clusters them by topic, detects agreements and contradictions using a neuro-symbolic hybrid stance classifier, and generates transparent, contradiction-aware summaries with 100% source attribution.

🌐 **Live Demo:** [https://newssummary-zzev4ju2gta656xhj4vtwa.streamlit.app/](https://newssummary-zzev4ju2gta656xhj4vtwa.streamlit.app/)

---

## 📌 Problem Statement & Novelty

When breaking news unfolds, multiple media outlets publish conflicting casualty figures, dates, and damage statistics. 

* **Standard AI Summarizers (ChatGPT / Gemini / T5):** Blend conflicting reports into a vague consensus (*"Between 47 and 100 people died..."*), concealing reporting errors and creating hallucinated compromises.
* **Our System:** Isolates individual factual claims, groups them by topic, and explicitly places contradictory assertions **side-by-side** with exact outlet names, timestamps, and reasoning. It does **not** pick a winner merely because more outlets repeated it.

---

## 🏗️ System Architecture

```
Multiple News Outlets (.txt / Pasted Text)
                    │
                    ▼
    [1. Atomic Claim Extraction & Normalization]
      Groq LLM (Qwen-27B) + Unit/Number Normalizer
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
      Rule Engine (Units, Disparities, Temporal Anchors)
      + Groq LLM Chain-of-Thought (5-Class FNC-1)
                    │
                    ▼
      [output/clusters_with_stance.json]
                    │
                    ▼
    [4. Contradiction-Aware Summary Synthesis]
      Categorization: Corroborated, Disputed, Developing, Unknowns
                    │
                    ▼
         [output/final_summary.md]
                    │
                    ▼
   [5. Interactive Streamlit Web Dashboard]
```

---

## 🚀 Quickstart

### 1. Prerequisites
- Python 3.10+
- Groq API Key ([console.groq.com](https://console.groq.com))

### 2. Setup Virtual Environment
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
Create a `.env` file in the project root:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
```

### 4. Launch the Interactive Dashboard
```bash
streamlit run app.py
```

---

## 💻 Running the Pipeline via CLI

You can also run each pipeline stage independently from the terminal:

```bash
# Step 1: Extract atomic factual claims
python src/01_extract_claims.py

# Step 2: Cluster claims and detect cross-source contradictions
python src/02_cluster_and_classify.py

# Step 3: Synthesize contradiction-aware summary report
python src/03_generate_summary.py

# Step 4: Run scientific evaluation benchmark
python src/evaluation.py
```

Generated artifacts:
- `output/claims.json` — Structured atomic claims with IDs, numerical values, and topics.
- `output/clusters_with_stance.json` — Semantic clusters and cross-source stance edges.
- `output/final_summary.md` — Complete publication report with side-by-side disputes.
- `output/evaluation_report.json` — Benchmark precision, recall, F1, and confusion matrix.

---

## 📊 Evaluation & Experimental Results

The stance classification engine was evaluated against a manually labeled 20-pair ground-truth benchmark (`data/evaluation_benchmark/benchmark_pairs.json`) across disaster, transportation, business, and edge cases:

| Stance Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **AGREE** | 1.00 | 1.00 | **1.00** | 5 |
| **DISAGREE** | 0.83 | 0.83 | **0.83** | 6 |
| **DISCUSS** | 0.50 | 1.00 | **0.67** | 4 |
| **UNRELATED** | 1.00 | 0.33 | **0.50** | 3 |
| **INSUFFICIENT_EVIDENCE** | 0.00 | 0.00 | **0.00** | 2 |
| **Overall Accuracy** | — | — | **75.0%** | 20 |
| **Macro Average** | 0.667 | 0.633 | **0.600** | — |

*Detailed confusion matrix and error analysis are documented in `output/evaluation_report.json`.*

---

## 📁 Repository Structure

```
Minor project 2026/
├── app.py                            # Streamlit Interactive Dashboard
├── requirements.txt                  # Python Dependencies
├── README.md                         # Project Guide
├── VIVA_GUIDE.md                     # Viva Q&A and Viva Presentation Guide
│
├── data/
│   ├── raw_sources/                  # Assam floods news corpus (4 outlets)
│   │   ├── README_contradiction_map.txt
│   │   ├── source1_wikipedia_compiled.txt
│   │   ├── source2_npr.txt
│   │   ├── source3_theunitedindian.txt
│   │   └── source4_mapsofindia.txt
│   ├── custom_sources/               # User uploaded / pasted news articles
│   └── evaluation_benchmark/         # Ground-truth evaluation dataset
│       └── benchmark_pairs.json
│
├── output/                           # Pipeline Output Artifacts
│   ├── claims.json                   # Extracted atomic claims
│   ├── clusters_with_stance.json     # Clustered facts & stance edges
│   ├── final_summary.md              # Synthesis report
│   └── evaluation_report.json        # Benchmark metrics & confusion matrix
│
├── src/                              # Core NLP Engine
│   ├── __init__.py                   # Package initializer
│   ├── config.py                     # Central configuration & paths
│   ├── extraction.py                 # Claim extraction engine
│   ├── clustering.py                 # Semantic clustering module
│   ├── stance_classifier.py          # Neuro-symbolic stance engine
│   ├── summarizer.py                 # Contradiction-aware synthesis
│   ├── evaluation.py                 # Benchmark testing script
│   ├── 01_extract_claims.py          # Backward-compatible Step 1 runner
│   ├── 02_cluster_and_classify.py    # Backward-compatible Step 2 runner
│   └── 03_generate_summary.py        # Backward-compatible Step 3 runner
│
└── generate_pdf_report.py            # Optional Academic PDF generator
```

---

## 🎓 Academic Credit & Viva Notes
Developed by **Aditya** as a **College Minor Project (2026)**.
For viva questions, model explanations, and conceptual defenses, refer to [VIVA_GUIDE.md](file:///d:/Minor%20project%202026/VIVA_GUIDE.md).
