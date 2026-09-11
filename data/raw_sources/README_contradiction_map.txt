EVENT CHOSEN: 2026 Assam Floods
WHY THIS EVENT: Multiple national, international, and regional-digital outlets reported
different death tolls and different "people affected" numbers for overlapping time periods,
with no single outlet fully explaining the discrepancy — exactly the scenario your project
targets (ground reports vs official statements vs wire copy).

FACT-SLOT 1: "Total death toll"
  - Source 1 (Wikipedia/ToI, 26 Jul): 66
  - Source 3 (TUI/ToI, 22 Jul):        "crossed 50"
  - Source 1 (ToI, 2 Aug):             82
  - Source 1 (compiled, 4 Aug):        85
  - Source 1 (compiled, 8 Aug):        98
  - Source 2 (NPR, 10 Aug):            100
  - Source 4 (MapsOfIndia, undated):   47
  => STANCE: Mostly "Discuss" (different dates = progression, not true contradiction)
     EXCEPT Source 4's "47" is a clear outlier/contradiction vs the same-period figures
     from Sources 1 and 2 — flag this pair for your stance classifier to test on.

FACT-SLOT 2: "People affected"
  - Source 1 (ToI, 2 Aug):        1.78 lakh (178,000)
  - Source 1 (compiled, 8 Aug):   155,000
  - Source 1 (compiled, 4 Aug):   "more than 1 lakh" (100,000+)
  - Source 4 (MapsOfIndia):       7.2 lakh (720,000)
  => STANCE: Source 4 vs Sources 1 is a strong DISAGREE pair — same rough time window,
     numbers differ by 4-6x. This is your best training/test example for Stage 4
     (Stance Classification).

FACT-SLOT 3: "Counting methodology" (meta-contradiction)
  - Source 3 explicitly states different outlets used different counting windows
    (full monsoon season vs confirmed-2026-only vs single 24-hour period).
  => This is gold for your report's "why aggregators fail" argument — one of your
     own sources basically confirms the problem statement in plain text.

HOW TO USE THIS FOR STEP 1 (Claim Extraction):
1. Feed each source1-4 .txt file separately into your LLM claim-extraction prompt.
2. Ask it to output atomic claims in JSON: {"claim": "...", "source": "source4", "date": "..."}
3. You should get ~5-8 claims per file.

HOW TO USE THIS FOR STEP 2 (Clustering):
- The two fact-slots above (death toll, people affected) should each form their own
  cluster automatically if your embedding model is working — use this as your
  first sanity check.

NEXT STEP (when you're ready): I can write you the actual claim-extraction prompt
to run on these 4 files.
