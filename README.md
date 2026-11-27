# News Contradiction Checker (Multi-Document News Analysis)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://newssummary-zzev4ju2gta656xhj4vtwa.streamlit.app/)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)

Compare multiple news articles on the same event side-by-side. Instead of blending conflicting numbers into a confusing average, this tool separates **conflicting facts**, **agreed facts**, **updates over time**, and **single-source claims** — always showing which news outlet said what.

🌐 **Live App:** https://newssummary-zzev4ju2gta656xhj4vtwa.streamlit.app/

---

## Why This Project Exists

When breaking news happens (such as floods, accidents, or policy changes), different news outlets often publish different numbers at the same time.

- **Normal AI Summarizers:** Blend conflicting reports into one vague line (*"Between 47 and 82 people died"*), hiding who reported which number.
- **This Tool:** Pulls out individual facts from each article, groups facts about the same topic, checks whether outlets agree or disagree, and shows conflicts **side-by-side** with exact source names and dates.

---

## How It Works (4 Simple Steps)

```text
News Articles (Pasted Text / .txt Upload / Built-in Sample)
                         │
                         ▼
   1. Pick Key Facts (src/extraction.py)
      Groq LLM (qwen/qwen3.8-27b) splits articles into single facts,
      converts Indian units ('1.78 lakh' -> 178000), and keeps dates & sources.
                         │
                         ▼
   2. Group Same Topics (src/clustering.py)
      Sentence-Transformers (all-MiniLM-L6-v2) + Agglomerative Clustering
      group facts talking about the same topic.
                         │
                         ▼
   3. Check Differences (src/stance_classifier.py)
      Rules (numbers, units, dates) + LLM reasoning label each pair:
      AGREE, DISAGREE, DISCUSS (date update), UNRELATED, or INSUFFICIENT_EVIDENCE.
                         │
                         ▼
   4. Show Side-by-Side (src/summarizer.py + app.py)
      Creates a short plain-English report and an interactive 3-tab web app.
```

---

## What You See in the Web App (`app.py`)

1. **📖 How It Works**
   - A quick, beginner-friendly overview of the 4 steps and the 4 result labels:
     - **🔴 Conflicting Facts:** Outlets give different numbers or facts for the same event.
     - **🟢 Agreed Facts:** Two or more outlets confirm the same fact.
     - **🟡 Later Updates:** Numbers changed across different reporting dates.
     - **⚪ Only 1 Source:** Mentioned by just one outlet so far.
2. **🔍 Compare & Results**
   - **Step 1:** Paste 2 news reports, upload `.txt` files, or run the built-in 4-report 2026 Assam Floods sample (also available in the sidebar).
   - **Step 2:** View summary counts, side-by-side **Conflict Cards**, **Agreed Facts**, **Updates Over Time** *(with `via ...` source attribution)*, and a short **Full Report** you can download as `.md`.
3. **📚 All Facts & Articles**
   - Search and filter every extracted fact by keyword, news source, or topic, and read the original `.txt` articles.

---

## Quickstart (Run Locally)

### 1. Requirements
- Python 3.11 (see [`runtime.txt`](runtime.txt))
- A free Groq API Key from [console.groq.com](https://console.groq.com)

### 2. Install
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

### 3. Add Your API Key
Copy [`.env.example`](.env.example) to `.env` in the project root:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
```
*(Or paste the key directly in the app sidebar at runtime. For Streamlit Cloud, see [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example).)*

### 4. Start the App
```bash
streamlit run app.py
```

---

## Run via Terminal & Tests

```bash
# Run each stage from the terminal
python src/extraction.py
python src/stance_classifier.py
python src/summarizer.py

# Run benchmark evaluation -> output/evaluation_report.json
python src/evaluation.py

# Run edge-case test suite (10 test cases)
python tests/test_edge_cases.py
```

---

## Evaluation Benchmark (`output/evaluation_report.json`)

Tested against 20 manually labeled claim pairs in [`data/evaluation_benchmark/benchmark_pairs.json`](data/evaluation_benchmark/benchmark_pairs.json):

| Label | Precision | Recall | F1-Score | Pairs |
|---|---|---|---|---|
| **AGREE** | 1.00 | 1.00 | **1.00** | 5 |
| **DISAGREE** | 0.83 | 0.83 | **0.83** | 6 |
| **DISCUSS** | 0.50 | 1.00 | **0.67** | 4 |
| **UNRELATED** | 1.00 | 0.33 | **0.50** | 3 |
| **INSUFFICIENT_EVIDENCE** | 0.00 | 0.00 | **0.00** | 2 |
| **Overall Accuracy** | — | — | **75.0%** | 20 |

---

## Project Files

```text
news_summary/
├── app.py                                # 3-tab Streamlit web app
├── requirements.txt                      # Python packages
├── runtime.txt                           # Python version (python-3.11)
├── .env.example                          # API key template
├── README.md                             # Project guide
│
├── .streamlit/
│   ├── config.toml                       # Streamlit theme & server settings
│   └── secrets.toml.example              # Streamlit Cloud secrets template
│
├── data/
│   ├── raw_sources/                      # Built-in 2026 Assam floods sample (4 articles)
│   ├── custom_sources/                   # Temporary folder for pasted/uploaded inputs
│   └── evaluation_benchmark/
│       └── benchmark_pairs.json          # 20 labeled test pairs
│
├── output/
│   ├── claims.json                       # Extracted facts
│   ├── clusters_with_stance.json         # Grouped facts & comparison results
│   ├── final_summary.md                  # Short plain-English report
│   └── evaluation_report.json            # Benchmark metrics
│
├── src/
│   ├── config.py                         # Paths, model settings, API key loader
│   ├── extraction.py                     # Step 1: Fact extraction & number normalization
│   ├── clustering.py                     # Step 2: Topic grouping using embeddings
│   ├── stance_classifier.py              # Step 3: Rules + LLM comparison engine
│   ├── summarizer.py                     # Step 4: Plain-English report generator
│   └── evaluation.py                     # Benchmark evaluation runner
│
└── tests/
    └── test_edge_cases.py                # 10-case edge & runtime test suite
```

---

## Author
Developed by **Aditya** (2026).
