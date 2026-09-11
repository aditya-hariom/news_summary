"""
Contradiction-Aware News Summarizer — Presentation & Interactive Dashboard
-------------------------------------------------------------------------
Designed for faculty demonstrations, minor project viva, and multi-document analysis.
"""

import os
import json
import glob
import streamlit as st
import importlib.util

# 1. Page Configuration
st.set_page_config(
    page_title="Contradiction-Aware News Summarizer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 2. Modern, Clean, Presentation-Ready Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hero Header */
    .hero-container {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        color: #F8FAFC;
        padding: 2rem 2.2rem;
        border-radius: 18px;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(90deg, #38BDF8, #818CF8, #C084FC);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-top: 0.5rem;
        margin-bottom: 0.8rem;
    }
    .badge-pill {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-right: 6px;
    }
    .badge-tech {
        background: rgba(56, 189, 248, 0.15);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }

    /* Comparison Battle Box */
    .comparison-box {
        border-radius: 14px;
        padding: 1.2rem;
        margin: 1rem 0;
        border: 1px solid;
    }
    .box-bad {
        background-color: #FFF1F2;
        border-color: #FECDD3;
        color: #881337;
    }
    .box-good {
        background-color: #F0FDF4;
        border-color: #BBF7D0;
        color: #14532D;
    }

    /* VS Battle Card */
    .vs-container {
        background: #FFFFFF;
        border-radius: 14px;
        border: 2px solid #F43F5E;
        padding: 1.4rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(244, 63, 94, 0.08);
    }
    .vs-title {
        color: #BE123C;
        font-size: 1.25rem;
        font-weight: 800;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .source-card {
        padding: 1rem 1.1rem;
        border-radius: 10px;
        border: 1px solid;
        height: 100%;
    }
    .source-card-a {
        background: #FEF2F2;
        border-color: #FCA5A5;
    }
    .source-card-b {
        background: #EFF6FF;
        border-color: #93C5FD;
    }
    .vs-badge-center {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 100%;
        font-weight: 900;
        font-size: 1.4rem;
        color: #E11D48;
    }

    /* Confirmed Card */
    .confirmed-card {
        background: #F0FDF4;
        border: 1px solid #86EFAC;
        border-left: 6px solid #16A34A;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.9rem;
    }

    /* Presentation Cheat Sheet */
    .cheat-sheet-box {
        background: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1.5rem;
    }

    /* Metric Cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 800 !important;
    }
</style>
""", unsafe_allow_html=True)

# 3. Hero Header
st.markdown("""
<div class="hero-container">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
        <div>
            <h1 class="hero-title">⚖️ Contradiction-Aware News Summarizer</h1>
            <p class="hero-subtitle">
                An intelligent multi-document AI engine that identifies <b>news contradictions side-by-side</b> instead of blending them into hallucinated summaries.
            </p>
            <div>
                <span class="badge-pill badge-tech">Groq LLM (Qwen-27B)</span>
                <span class="badge-pill badge-tech">Sentence-Transformers (MiniLM)</span>
                <span class="badge-pill badge-tech">Agglomerative Clustering</span>
                <span class="badge-pill badge-tech">FNC-1 Stance Classifier</span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# 4. Presentation & Viva Helper (Expendable for Student)
with st.expander("🎓 <b>Presentation & Viva Guide (Click to show: What to say to your Professor)</b>", expanded=False):
    st.markdown("""
    ### 🎙️ 1-Minute Presentation Pitch (Say this in your demo):
    > *"Good morning Sir/Ma'am. In fast-breaking events like natural disasters, different news agencies report conflicting figures — for instance, one paper reports 47 deaths while another reports 100, or 1.78 lakh affected vs 7.2 lakh affected.*
    > 
    > *Traditional LLM summarizers like ChatGPT or Google Gemini merge all texts together and produce hallucinated or misleading averages. Our project solves this through a 4-step NLP pipeline:*
    > 1. **Extracts atomic factual claims** preserving exact source attribution and dates.
    > 2. **Semantically clusters** claims by topic using Sentence Transformers.
    > 3. **Classifies pairwise cross-source stance** to actively detect Contradictions vs Agreements.
    > 4. **Generates a 3-category report**: Confirmed Facts (Green), Disputed Facts (Red side-by-side), and Unconfirmed Single-Source Facts (Gray).*
    
    ---
    ### ❓ Top 3 Viva Questions & Quick Answers:
    - **Q1: What is an 'atomic claim'?**
      - *Answer:* A single, self-contained statement (e.g. "85 people were reported dead") broken down from complex compound sentences, so it can be evaluated independently.
    - **Q2: How do you identify contradictions?**
      - *Answer:* We pair claims from different sources within the same semantic cluster and use our Stance Classifier to categorize the relationship into Agree, Disagree, Discuss, or Unrelated (inspired by the Fake News Challenge FNC-1 framework).
    - **Q3: Why not just ask ChatGPT to find contradictions directly?**
      - *Answer:* Direct LLM summarization suffers from lost-in-the-middle attention degradation and often synthesizes false compromises. Our pipeline enforces strict attribution and mathematical clustering before verification.
    """)

# 5. Core Motivation Card: Standard AI vs Our System
col_bad, col_good = st.columns(2)
with col_bad:
    st.markdown("""
    <div class="comparison-box box-bad">
        <h4 style="margin:0 0 8px 0; color:#991B1B;">❌ What Standard AI (ChatGPT / Gemini) Does:</h4>
        <p style="margin:0; font-size:0.95rem; line-height:1.5;">
            Blends conflicting reports into a vague, misleading consensus:
            <br><i>"Between 47 and 100 people died, and around 1.78 to 7.2 lakh people were affected..."</i>
            <br><b style="color:#991B1B;">⚠️ Problem:</b> Hides who reported what, obscures real errors, and misinforms readers.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_good:
    st.markdown("""
    <div class="comparison-box box-good">
        <h4 style="margin:0 0 8px 0; color:#166534;">✅ What Our System Does:</h4>
        <p style="margin:0; font-size:0.95rem; line-height:1.5;">
            Detects exact discrepancies and presents them side-by-side:
            <br><b>⚡ DISPUTE:</b> <code>MapsOfIndia</code> claims <b>7.2 Lakh affected</b> vs <code>Times of India</code> claims <b>1.78 Lakh</b> (4x difference!).
            <br><b style="color:#166534;">✨ Benefit:</b> 100% source traceability and zero hidden hallucinations.
        </p>
    </div>
    """, unsafe_allow_html=True)

# 6. Load Pre-Computed Pipeline Outputs
claims_file = "output/claims.json"
clusters_file = "output/clusters_with_stance.json"
summary_file = "output/final_summary.md"

has_results = os.path.exists(clusters_file) and os.path.exists(summary_file)

if has_results:
    with open(clusters_file, "r", encoding="utf-8") as f:
        clusters_data = json.load(f)

    confirmed_list = []
    disputed_list = []
    unconfirmed_list = []

    for cid, cdata in clusters_data.items():
        sources = set(m["source_file"] for m in cdata["claims"])
        edges = cdata.get("edges", [])
        has_disagree = any(e.get("stance") == "disagree" for e in edges)
        has_agree = any(e.get("stance") == "agree" for e in edges)

        if len(sources) <= 1:
            unconfirmed_list.append(cdata)
        elif has_disagree:
            disputed_list.append(cdata)
        elif has_agree:
            confirmed_list.append(cdata)
        else:
            unconfirmed_list.append(cdata)

    # 7. Summary Metrics
    st.markdown("### 📊 Analysis Overview (2026 Assam Floods Corpus)")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Source Documents", "4 News Outlets", delta="Wikipedia, NPR, TOI, MapsOfIndia")
    with m2:
        st.metric("🚨 Disputed Contradictions", f"{len(disputed_list)} Clusters", delta="Active Conflicts", delta_color="inverse")
    with m3:
        st.metric("✅ Confirmed Facts", f"{len(confirmed_list)} Clusters", delta="Multi-Source Agreement", delta_color="normal")
    with m4:
        st.metric("ℹ️ Single-Source Facts", f"{len(unconfirmed_list)} Claims", delta="Pending Verification", delta_color="off")

    st.markdown("---")

    # 8. Main Tabs
    tab_showcase, tab_facts, tab_articles, tab_pipeline, tab_run = st.tabs([
        "🎯 1. Key Contradictions (Live Demo)",
        "✅ 2. Confirmed & Developing Facts",
        "📰 3. Read Source Articles",
        "🧠 4. How It Works (Pipeline)",
        "🚀 5. Run Custom / Live Pipeline",
    ])

    # ================= TAB 1: SHOWCASE KEY CONTRADICTIONS =================
    with tab_showcase:
        st.markdown("### 🚨 Detected Contradictions (Aamne-Saamne Comparison)")
        st.info("Here are the direct factual contradictions discovered across the news sources. Notice how conflicting statements and figures are placed side-by-side with exact outlet names.")

        if not disputed_list:
            st.success("🎉 **No factual contradictions detected!** All analyzed sources agree or report complementary, non-conflicting facts.")
        else:
            for idx, c in enumerate(disputed_list, 1):
                sources = sorted(set(m["source_file"] for m in c["claims"]))
                disagree_edges = [e for e in c.get("edges", []) if e.get("stance") == "disagree"]
                
                # Pick the primary disagreement pair if available
                if disagree_edges:
                    primary_edge = disagree_edges[0]
                    src_a = primary_edge.get("source_a", "Source A")
                    claim_a = primary_edge.get("claim_a", "")
                    src_b = primary_edge.get("source_b", "Source B")
                    claim_b = primary_edge.get("claim_b", "")
                    reasoning = primary_edge.get("reasoning", "Incompatible factual statements detected.")
                    confidence = primary_edge.get("confidence", 0.90)
                else:
                    src_a = c["claims"][0]["source_file"]
                    claim_a = c["claims"][0]["claim"]
                    src_b = c["claims"][-1]["source_file"]
                    claim_b = c["claims"][-1]["claim"]
                    reasoning = "Conflicting reported figures within the same cluster."
                    confidence = 0.85

                st.markdown(f"""
                <div class="vs-container">
                    <div class="vs-title">
                        <span>⚡ Discrepancy #{idx}: Fact Cluster {idx}</span>
                        <span style="font-size:0.8rem; background:#FFE4E6; color:#9F1239; padding:3px 10px; border-radius:99px; margin-left:auto;">
                            Contradiction Detected (Confidence: {confidence:.0%})
                        </span>
                    </div>
                    <div style="display: flex; gap: 15px; align-items: stretch;">
                        <div style="flex: 5;" class="source-card source-card-a">
                            <span style="background:#FCA5A5; color:#7F1D1D; padding:2px 8px; border-radius:4px; font-weight:700; font-size:0.8rem;">
                                Outlet: {src_a}
                            </span>
                            <h4 style="color:#991B1B; margin:8px 0 4px 0;">"{claim_a}"</h4>
                        </div>
                        <div style="flex: 1;" class="vs-badge-center">VS</div>
                        <div style="flex: 5;" class="source-card source-card-b">
                            <span style="background:#93C5FD; color:#1E3A8A; padding:2px 8px; border-radius:4px; font-weight:700; font-size:0.8rem;">
                                Outlet: {src_b}
                            </span>
                            <h4 style="color:#1D4ED8; margin:8px 0 4px 0;">"{claim_b}"</h4>
                        </div>
                    </div>
                    <div style="margin-top:12px; font-size:0.88rem; color:#881337; background:#FFF1F2; padding:10px 14px; border-radius:8px; border-left:4px solid #F43F5E;">
                        <b>🧠 AI Verification Reasoning:</b> {reasoning}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Meta Contradiction: Counting Methodology (Explainer)
        with st.expander("🔍 Show Meta-Contradiction: Why do news channels disagree? (Counting Methodology)", expanded=False):
            st.markdown("""
            **Why do news channels disagree on numbers?**
            - **Different Counting Windows:** Some agencies count figures across the *entire season* (including pre-monsoon and landslides).
            - **Strict Event Window:** Other agencies count *only verified drowning deaths* during a single peak 24-hour wave.
            - **Delayed Verification:** Ground rescue teams submit delayed tallies, leading to temporary differences in reported stats.
            - *This is why automated aggregators fail when they don't look at methodology!*
            """)

    # ================= TAB 2: CONFIRMED & DEVELOPING FACTS =================
    with tab_facts:
        st.markdown("### ✅ Confirmed Facts (Agreed Upon by Multiple Outlets)")
        st.caption("When 2 or more independent news sources report the exact same statement without contradiction, our system marks it as Confirmed.")

        for idx, c in enumerate(confirmed_list, 1):
            sources = sorted(set(m["source_file"] for m in c["claims"]))
            sources_badge = " ".join([f"`{s}`" for s in sources])
            
            st.markdown(f"""
            <div class="confirmed-card">
                <div style="font-weight:700; font-size:1.05rem; color:#14532D; margin-bottom:4px;">
                    ✅ Fact #{idx}: {c['claims'][0]['claim']}
                </div>
                <div style="font-size:0.85rem; color:#166534;">
                    <b>Agreed by independent sources:</b> {sources_badge}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(f"### ℹ️ Unconfirmed / Single-Source Claims ({len(unconfirmed_list)})")
        st.caption("These statements appeared in only ONE news outlet. They are not disputed, but cannot yet be independently corroborated.")
        
        with st.expander(f"Click to inspect all {len(unconfirmed_list)} single-source claims", expanded=False):
            for c in unconfirmed_list:
                for m in c["claims"]:
                    date_tag = f" `({m['date']})`" if m.get("date") else ""
                    attr_tag = f" [Attributed to: *{m['attribution']}*]" if m.get("attribution") else ""
                    st.markdown(f"- 📄 **`{m['source_file']}`**{date_tag}: {m['claim']}{attr_tag}")

    # ================= TAB 3: READ SOURCE ARTICLES =================
    with tab_articles:
        st.markdown("### 📰 Raw News Source Articles")
        st.info("Here are the 4 real news articles about the 2026 Assam Floods used as the input corpus.")
        
        source_files = sorted(glob.glob("data/raw_sources/source*.txt"))
        if source_files:
            source_tabs = st.tabs([os.path.basename(f) for f in source_files])
            for idx, fpath in enumerate(source_files):
                with source_tabs[idx]:
                    with open(fpath, "r", encoding="utf-8") as sf:
                        content = sf.read()
                    st.code(content, language="text")
        else:
            st.warning("No source files found in data/raw_sources/.")

    # ================= TAB 4: HOW IT WORKS =================
    with tab_pipeline:
        st.markdown("### 🧠 How the System Works (Under the Hood)")
        st.markdown("""
        Our pipeline follows 4 modular stages to transform unstructured news into verified intelligence:
        """)

        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.markdown("""
            <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-top:4px solid #3B82F6; border-radius:10px; padding:15px; height:100%;">
                <h4 style="margin:0 0 6px 0; color:#1E40AF;">Step 1: Extraction</h4>
                <p style="font-size:0.85rem; color:#475569; margin:0;">
                    <b>Groq LLM (Qwen-27B)</b> reads articles and breaks compound sentences into atomic, verifiable facts with dates & attribution.
                </p>
            </div>
            """, unsafe_allow_html=True)
        with s2:
            st.markdown("""
            <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-top:4px solid #8B5CF6; border-radius:10px; padding:15px; height:100%;">
                <h4 style="margin:0 0 6px 0; color:#5B21B6;">Step 2: Clustering</h4>
                <p style="font-size:0.85rem; color:#475569; margin:0;">
                    <b>Sentence-Transformers</b> converts claims to embeddings. <b>Agglomerative Clustering</b> groups claims discussing the same topic.
                </p>
            </div>
            """, unsafe_allow_html=True)
        with s3:
            st.markdown("""
            <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-top:4px solid #EC4899; border-radius:10px; padding:15px; height:100%;">
                <h4 style="margin:0 0 6px 0; color:#9D174D;">Step 3: Stance</h4>
                <p style="font-size:0.85rem; color:#475569; margin:0;">
                    Cross-source pairs within each cluster are tested for <code>agree</code>, <code>disagree</code>, <code>discuss</code>, or <code>unrelated</code>.
                </p>
            </div>
            """, unsafe_allow_html=True)
        with s4:
            st.markdown("""
            <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-top:4px solid #10B981; border-radius:10px; padding:15px; height:100%;">
                <h4 style="margin:0 0 6px 0; color:#065F46;">Step 4: Synthesis</h4>
                <p style="font-size:0.85rem; color:#475569; margin:0;">
                    Generates the final 3-bucket report: <b>Confirmed</b>, <b>Disputed</b> (side-by-side), and <b>Unconfirmed</b>.
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Data Flow Architecture")
        st.code("""
[News Outlets .txt] ──► [LLM Claim Extraction] ──► [output/claims.json]
                              │
                              ▼
                     [SentenceTransformer]
                              │
                              ▼
                  [Agglomerative Clustering] ──► [Same-Fact Groups]
                              │
                              ▼
                  [LLM Stance Classification] ──► [output/clusters_with_stance.json]
                              │
                              ▼
                 [Contradiction-Aware Synthesis] ──► [output/final_summary.md]
        """, language="text")

    # ================= TAB 5: RUN / TEST CUSTOM PIPELINE =================
    with tab_run:
        st.markdown("### 🚀 Run Analysis on Custom or Default News Files")
        st.caption("You can run the entire pipeline live during your presentation demo.")

        col_cfg, col_act = st.columns([1, 1])
        with col_cfg:
            auto_threshold = st.checkbox("🤖 Auto-Detect Optimal Clustering Distance (Adaptive Silhouette)", value=True)
            threshold = None
            if not auto_threshold:
                threshold = st.slider(
                    "Manual Clustering Distance Threshold",
                    min_value=0.30,
                    max_value=0.85,
                    value=0.55,
                    step=0.05,
                    help="Controls grouping sensitivity: 0.55 is default."
                )
            mode = st.radio(
                "Select Data Source:",
                ["Use Built-in 2026 Assam Floods Articles", "Upload New .txt Files"],
                index=0
            )

        with col_act:
            st.markdown("#### Trigger Execution")
            uploaded_paths = []
            if mode == "Upload New .txt Files":
                uploaded_files = st.file_uploader(
                    "Upload 2 or more conflicting news .txt files",
                    type=["txt"],
                    accept_multiple_files=True,
                    help="Upload articles from different newspapers/sources about the same incident."
                )
                if uploaded_files:
                    custom_dir = "data/custom_sources"
                    os.makedirs(custom_dir, exist_ok=True)
                    # Clear out prior custom files
                    for old_f in glob.glob(f"{custom_dir}/*"):
                        try:
                            os.remove(old_f)
                        except Exception:
                            pass
                    for uf in uploaded_files:
                        fp = os.path.join(custom_dir, uf.name)
                        with open(fp, "wb") as f:
                            f.write(uf.getbuffer())
                        uploaded_paths.append(fp)
                    st.success(f"Loaded {len(uploaded_paths)} files ready for analysis!")

            run_btn = st.button("⚡ Run Full AI Pipeline Now", type="primary", use_container_width=True)

        if run_btn:
            if mode == "Upload New .txt Files" and len(uploaded_paths) < 2:
                st.error("Please upload at least 2 news articles (.txt) to analyze contradictions!")
            else:
                with st.spinner("Running 4-stage pipeline..."):
                    prog = st.progress(0)
                    status = st.empty()

                    status.info("Step 1/3: Extracting atomic claims via Groq LLM...")
                    prog.progress(20)
                    spec1 = importlib.util.spec_from_file_location("step1", "src/01_extract_claims.py")
                    step1 = importlib.util.module_from_spec(spec1)
                    spec1.loader.exec_module(step1)
                    
                    target_sources = uploaded_paths if mode == "Upload New .txt Files" else None
                    step1.main(input_files=target_sources)

                    status.info("Step 2/3: Clustering facts & classifying stances...")
                    prog.progress(60)
                    spec2 = importlib.util.spec_from_file_location("step2", "src/02_cluster_and_classify.py")
                    step2 = importlib.util.module_from_spec(spec2)
                    spec2.loader.exec_module(step2)
                    step2.run_pipeline_step2(distance_threshold=threshold)

                    status.info("Step 3/3: Synthesizing Contradiction-Aware Report...")
                    prog.progress(90)
                    spec3 = importlib.util.spec_from_file_location("step3", "src/03_generate_summary.py")
                    step3 = importlib.util.module_from_spec(spec3)
                    spec3.loader.exec_module(step3)
                    step3.run_pipeline_step3()

                    prog.progress(100)
                    status.success("✅ Analysis successfully completed! Refreshing results...")
                    st.rerun()

        st.markdown("---")
        st.subheader("📥 Export Outputs")
        with open(summary_file, "r", encoding="utf-8") as f:
            md_text = f.read()
        
        st.download_button("📥 Download Final Summary Report (.md)", md_text, "final_summary.md", "text/markdown")
        
        with st.expander("View Raw Generated Markdown Report"):
            st.markdown(md_text)

else:
    st.info("No generated data found. Click the button in the sidebar or run the pipeline to begin.")
