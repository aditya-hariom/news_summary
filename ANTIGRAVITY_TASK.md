# Instructions for Antigravity Agent

Paste this whole file's content (or just point to it) as your task to the Antigravity
agent when you open this project folder.

## Project
"Contradiction-Aware Multi-Document Summarizer" — a minor project that takes multiple
conflicting news source files about one event and outputs a structured summary that
separates Confirmed facts from Disputed facts, instead of blending them into one narrative.

## What's already here
- `data/raw_sources/` — 4 real news-source text files about the 2026 Assam Floods,
  plus a README_contradiction_map.txt explaining the known contradictions.
- `src/01_extract_claims.py` — Step 1: extracts atomic factual claims from each source file via LLM.
- `src/02_cluster_and_classify.py` — Step 2: clusters claims about the same fact, then
  classifies each cross-source pair as agree/disagree/discuss/unrelated (FNC-1 style).
- `src/03_generate_summary.py` — Step 3: renders the final Confirmed/Disputed/Unconfirmed
  markdown report from the clustered+classified data.
- `requirements.txt`

## Task for you (Antigravity)
1. Set up a Python virtual environment and `pip install -r requirements.txt`.
2. Check that an `ANTHROPIC_API_KEY` environment variable is set. If not, tell me to set
   it (do not hardcode a key in any file).
3. Run the pipeline in order:
   ```
   python src/01_extract_claims.py
   python src/02_cluster_and_classify.py
   python src/03_generate_summary.py
   ```
4. After each step, print/show me the output file content so I can sanity-check it
   (output/claims.json, then output/clusters_with_stance.json, then output/final_summary.md).
5. If `01_extract_claims.py` fails to parse JSON from the model for any source file,
   fix the prompt or add retry logic — don't silently skip a file.
6. If clustering in step 2 puts unrelated claims in the same cluster, or splits the
   same fact into two clusters, try adjusting `distance_threshold` in
   `cluster_claims()` (currently 0.55) and re-run — show me before/after cluster counts.
7. Once the pipeline runs end-to-end cleanly, build a minimal Streamlit app
   (`src/04_app.py`) with:
   - A file uploader for multiple .txt source files
   - A "Run Analysis" button that calls the 3 pipeline steps in sequence
   - Renders `output/final_summary.md` on screen with Confirmed facts in green,
     Disputed facts in red/highlighted, Unconfirmed in gray
   This is the part I'll demo to my professor, so it should look clean, not raw JSON.

## Things to NOT do
- Don't invent new source files or fabricate news content — only use what's in
  `data/raw_sources/`.
- Don't remove the source attribution from claims anywhere in the pipeline — the whole
  point of the project is traceability back to which outlet said what.
- Don't merge disputed claims into a single sentence in the final summary — they must
  stay side by side with attribution.

## After it's working
Ask me whether I want to:
(a) add more source files / a second event as a second test case, or
(b) write the evaluation section (testing the stance classifier against a small
    hand-labeled sample) for the project report.
