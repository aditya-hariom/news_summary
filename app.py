"""
Contradiction-Aware News Summarizer — Presentation & Interactive Dashboard
-------------------------------------------------------------------------
Designed for faculty demonstrations, minor project viva, and multi-document news analysis.
Supports direct text pasting, file uploads, claims exploration, and side-by-side contradiction tracking.
"""

import os
import sys
import json
import glob
import streamlit as st
from dotenv import load_dotenv

# Ensure project root is in sys.path for direct imports
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    get_groq_api_key,
    GROQ_MODEL,
    EMBEDDING_MODEL,
    CLAIMS_FILE,
    CLUSTERS_FILE,
    SUMMARY_FILE,
    RAW_SOURCES_DIR,
    CUSTOM_SOURCES_DIR,
)
from src.extraction import (
    extract_claims_from_text,
    run_extraction_pipeline,
)
from src.stance_classifier import run_stance_pipeline
from src.summarizer import run_summary_pipeline

# 1. Page Configuration
st.set_page_config(
    page_title="Contradiction-Aware News Summarizer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Modern, Presentation-Ready Styling
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
        font-size: 1.2rem;
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

    /* Claim Tag */
    .claim-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# 3. Hero Header
st.markdown("""
<div class="hero-container">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
        <div>
            <h1 class="hero-title">⚖️ Contradiction-Aware News Analysis System</h1>
            <p class="hero-subtitle">
                An intelligent multi-document AI engine that extracts atomic claims, clusters factual topics, and surfaces <b>news contradictions side-by-side</b> with 100% source attribution.
            </p>
            <div>
                <span class="badge-pill badge-tech">Groq LLM (Qwen-27B)</span>
                <span class="badge-pill badge-tech">Sentence-Transformers (MiniLM-L6)</span>
                <span class="badge-pill badge-tech">Agglomerative Clustering</span>
                <span class="badge-pill badge-tech">Neuro-Symbolic Stance Engine</span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# 4. Sidebar: System Controls & API Status
with st.sidebar:
    st.markdown("### ⚙️ System Controls")
    active_key = get_groq_api_key()
    if active_key:
        st.success("🟢 **Groq API**: Connected")
    else:
        st.warning("🟡 **Groq API**: Missing")
        user_key = st.text_input("Enter Groq API Key:", type="password", placeholder="gsk_...")
        if user_key:
            os.environ["GROQ_API_KEY"] = user_key.strip()
            st.success("API key registered for session!")
            st.rerun()

    st.markdown("---")
    st.markdown("### 📋 Viva Reference Card")
    st.markdown(f"- **LLM:** `{GROQ_MODEL}`")
    st.markdown(f"- **Embedder:** `{EMBEDDING_MODEL}`")
    st.markdown("- **Clustering:** Agglomerative (Cosine)")
    st.markdown("- **Stance Scheme:** 5-Class FNC-1 Extended")
    st.markdown("- **Threshold Search:** Adaptive Silhouette")
    st.markdown("---")
    st.caption("College Minor Project 2026 — Aditya")

# 5. Motivation Banner (Standard AI vs Our System)
col_bad, col_good = st.columns(2)
with col_bad:
    st.markdown("""
    <div class="comparison-box box-bad">
        <h4 style="margin:0 0 6px 0; color:#991B1B;">❌ What Standard AI (ChatGPT / Gemini) Does:</h4>
        <p style="margin:0; font-size:0.9rem; line-height:1.4;">
            Blends conflicting reports into a vague, misleading consensus:
            <br><i>"Between 47 and 100 people died, and 1.78 to 7.2 lakh people were affected..."</i>
            <br><b style="color:#991B1B;">⚠️ Consequence:</b> Conceals discrepancies, obscures errors, and hallucinated false compromises.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_good:
    st.markdown("""
    <div class="comparison-box box-good">
        <h4 style="margin:0 0 6px 0; color:#166534;">✅ What Our System Does:</h4>
        <p style="margin:0; font-size:0.9rem; line-height:1.4;">
            Isolates factual claims and displays discrepancies side-by-side:
            <br><b>⚡ DISPUTE:</b> <code>MapsOfIndia</code> reports <b>7.2 Lakh affected</b> vs <code>Times of India</code> reports <b>1.78 Lakh</b> (4x disparity!).
            <br><b style="color:#166534;">✨ Benefit:</b> Preserves exact source attribution, temporal context, and zero hidden edits.
        </p>
    </div>
    """, unsafe_allow_html=True)

# 6. Load Pre-Computed Pipeline Outputs
has_results = os.path.exists(CLUSTERS_FILE) and os.path.exists(CLAIMS_FILE) and os.path.exists(SUMMARY_FILE)

if has_results:
    with open(CLAIMS_FILE, "r", encoding="utf-8") as f:
        claims_data = json.load(f)

    with open(CLUSTERS_FILE, "r", encoding="utf-8") as f:
        clusters_data = json.load(f)

    with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
        summary_markdown = f.read()

    # Dynamic metrics calculation
    all_sources = sorted(set(c.get("source_name") or c.get("source_file") for c in claims_data))
    
    disputed_clusters = []
    corroborated_clusters = []
    unconfirmed_clusters = []
    total_disagreements = 0

    for cid, cdata in clusters_data.items():
        claims = cdata["claims"]
        edges = cdata.get("edges", [])
        sources = set(c.get("source_name") or c.get("source_file") for c in claims)
        has_disagree = any(e.get("stance") == "DISAGREE" for e in edges)
        has_agree = any(e.get("stance") == "AGREE" for e in edges)

        disagree_count = sum(1 for e in edges if e.get("stance") == "DISAGREE")
        total_disagreements += disagree_count

        if len(sources) <= 1:
            unconfirmed_clusters.append(cdata)
        elif has_disagree:
            disputed_clusters.append(cdata)
        elif has_agree:
            corroborated_clusters.append(cdata)
        else:
            unconfirmed_clusters.append(cdata)

    # 7. Summary Metrics
    st.markdown("### 📊 Active Corpus Overview")
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Source Articles", f"{len(all_sources)} Outlets", delta=f"{len(claims_data)} Claims")
    with m2:
        st.metric("Fact Clusters", f"{len(clusters_data)} Clusters", delta="Semantic Topics")
    with m3:
        st.metric("🚨 Active Contradictions", f"{len(disputed_clusters)} Clusters", delta=f"{total_disagreements} Disagree Pairs", delta_color="inverse")
    with m4:
        st.metric("✅ Corroborated Facts", f"{len(corroborated_clusters)} Topics", delta="Multi-Source Agree", delta_color="normal")
    with m5:
        st.metric("ℹ️ Single-Source Facts", f"{len(unconfirmed_clusters)} Topics", delta="Pending Verification", delta_color="off")

    st.markdown("---")

    # 8. Main Interactive Tabs
    tab_contra, tab_corrob, tab_claims, tab_summary, tab_articles, tab_architecture, tab_run = st.tabs([
        "🎯 1. Contradiction Explorer",
        "✅ 2. Corroborated & Developing",
        "🔍 3. Claims Explorer",
        "📄 4. Structured Summary",
        "📰 5. Source Articles",
        "🧠 6. Pipeline Architecture",
        "🚀 7. Run Live Analysis",
    ])

    # ================= TAB 1: CONTRADICTION EXPLORER =================
    with tab_contra:
        st.markdown("### 🚨 Detected News Contradictions (Side-by-Side Perspectives)")
        st.caption("These cross-source discrepancies were detected within the same factual topic cluster. Notice how figures and dates are juxtaposed without merging.")

        if not disputed_clusters:
            st.success("🎉 No factual contradictions detected across the active sources.")
        else:
            for idx, c in enumerate(disputed_clusters, 1):
                topic = c.get("topic", "General Incident").replace("_", " ").title()
                disagree_edges = [e for e in c.get("edges", []) if e.get("stance") == "DISAGREE"]
                
                # Show each unique contradiction edge
                seen_pairs = set()
                for e_idx, edge in enumerate(disagree_edges, 1):
                    pair_key = tuple(sorted([edge["claim_a"], edge["claim_b"]]))
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)

                    src_a = edge.get("source_a", "Source A")
                    claim_a = edge.get("claim_a", "")
                    date_a = f" ({edge['date_a']})" if edge.get("date_a") else ""
                    val_a_str = f" [Value: {edge['val_a']} {edge.get('unit_a','')}]" if edge.get("val_a") is not None else ""

                    src_b = edge.get("source_b", "Source B")
                    claim_b = edge.get("claim_b", "")
                    date_b = f" ({edge['date_b']})" if edge.get("date_b") else ""
                    val_b_str = f" [Value: {edge['val_b']} {edge.get('unit_b','')}]" if edge.get("val_b") is not None else ""

                    attr = edge.get("conflicting_attribute") or topic
                    conf = edge.get("confidence", 0.95)
                    reasoning = edge.get("reasoning", "Incompatible factual reports detected.")

                    st.markdown(f"""
                    <div class="vs-container">
                        <div class="vs-title">
                            <span>⚡ Contradiction #{idx}.{e_idx}: {attr} Conflict</span>
                            <span style="font-size:0.8rem; background:#FFE4E6; color:#9F1239; padding:3px 10px; border-radius:99px; margin-left:auto;">
                                Discrepancy Verified (Confidence: {conf:.0%})
                            </span>
                        </div>
                        <div style="display: flex; gap: 15px; align-items: stretch;">
                            <div style="flex: 5;" class="source-card source-card-a">
                                <span style="background:#FCA5A5; color:#7F1D1D; padding:2px 8px; border-radius:4px; font-weight:700; font-size:0.8rem;">
                                    Outlet: {src_a}{date_a}
                                </span>
                                <h4 style="color:#991B1B; margin:8px 0 4px 0;">"{claim_a}"</h4>
                                <div style="font-size:0.82rem; color:#7F1D1D;">{val_a_str}</div>
                            </div>
                            <div style="flex: 1;" class="vs-badge-center">VS</div>
                            <div style="flex: 5;" class="source-card source-card-b">
                                <span style="background:#93C5FD; color:#1E3A8A; padding:2px 8px; border-radius:4px; font-weight:700; font-size:0.8rem;">
                                    Outlet: {src_b}{date_b}
                                </span>
                                <h4 style="color:#1D4ED8; margin:8px 0 4px 0;">"{claim_b}"</h4>
                                <div style="font-size:0.82rem; color:#1E3A8A;">{val_b_str}</div>
                            </div>
                        </div>
                        <div style="margin-top:12px; font-size:0.88rem; color:#881337; background:#FFF1F2; padding:10px 14px; border-radius:8px; border-left:4px solid #F43F5E;">
                            <b>🧠 AI Verification Reasoning:</b> {reasoning}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        with st.expander("🔍 Show Meta-Contradiction: Why do news outlets disagree? (Methodology Gap)", expanded=False):
            st.markdown("""
            **Why do news channels report conflicting figures?**
            1. **Different Counting Windows:** Some agencies count seasonal totals (including landslides/disease); others count only verified drowning deaths during a single 24-hour wave.
            2. **Official vs Ground Reports:** Wire agencies quote provisional ground numbers, while government bulletins publish delayed verified figures.
            3. **Lack of Retractions:** Outlets rarely issue formal retractions when old estimates are superseded.
            """)

    # ================= TAB 2: CORROBORATED & DEVELOPING =================
    with tab_corrob:
        st.markdown("### ✅ Corroborated Information (Independent Multi-Source Agreement)")
        st.caption("When 2 or more independent news outlets publish matching statements with zero detected contradictions, our system flags them as corroborated.")

        for idx, c in enumerate(corroborated_clusters, 1):
            sources = sorted(set(m.get("source_name") or m.get("source_file") for m in c["claims"]))
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
        st.markdown(f"### ℹ️ Single-Source Claims ({len(unconfirmed_clusters)} Topics)")
        st.caption("These statements appeared in only ONE news outlet. They are not disputed, but remain uncorroborated.")
        with st.expander("Inspect all single-source claims", expanded=False):
            for c in unconfirmed_clusters:
                for m in c["claims"]:
                    src = m.get("source_name") or m.get("source_file")
                    d_str = f" `({m['date']})`" if m.get("date") else ""
                    attr_str = f" [Attributed to: *{m['attribution']}*]" if m.get("attribution") else ""
                    st.markdown(f"- 📄 **`{src}`**{d_str}: \"{m['claim']}\"{attr_str}")

    # ================= TAB 3: CLAIMS EXPLORER =================
    with tab_claims:
        st.markdown("### 🔍 Atomic Claims Explorer")
        st.caption("Search and filter all granular factual statements extracted by the LLM.")

        col_search, col_src_filter, col_cat_filter = st.columns([2, 1, 1])
        with col_search:
            search_query = st.text_input("🔎 Search Claims by keyword:", placeholder="e.g. death, affected, river, rescue")
        with col_src_filter:
            selected_src = st.selectbox("Filter by Source:", ["All Sources"] + all_sources)
        with col_cat_filter:
            all_cats = sorted(set(c.get("category") or c.get("topic") or "general" for c in claims_data))
            selected_cat = st.selectbox("Filter by Category:", ["All Categories"] + all_cats)

        filtered_claims = claims_data
        if search_query:
            filtered_claims = [c for c in filtered_claims if search_query.lower() in c.get("claim", "").lower()]
        if selected_src != "All Sources":
            filtered_claims = [c for c in filtered_claims if (c.get("source_name") or c.get("source_file")) == selected_src]
        if selected_cat != "All Categories":
            filtered_claims = [c for c in filtered_claims if (c.get("category") or c.get("topic")) == selected_cat]

        st.info(f"Showing **{len(filtered_claims)}** matching atomic claims:")

        for c in filtered_claims:
            cid = c.get("claim_id", "N/A")
            src = c.get("source_name") or c.get("source_file")
            topic = c.get("category") or c.get("topic", "general")
            val_str = f" | **Value:** `{c['numerical_value']} {c.get('numerical_unit', '')}`" if c.get("numerical_value") is not None else ""
            date_str = f" | **Date:** `{c['date']}`" if c.get("date") else ""
            attr_str = f" | **Attribution:** *{c['attribution']}*" if c.get("attribution") else ""

            st.markdown(f"""
            <div class="claim-box">
                <div style="font-size:0.8rem; color:#64748B; margin-bottom:4px;">
                    <code>{cid}</code> • <b>Outlet:</b> <code>{src}</code> • <b>Topic:</b> <span style="background:#E2E8F0; padding:1px 6px; border-radius:4px;">{topic}</span>
                    {val_str}{date_str}{attr_str}
                </div>
                <div style="font-size:0.95rem; font-weight:600; color:#0F172A;">
                    "{c.get('claim')}"
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ================= TAB 4: STRUCTURED SUMMARY =================
    with tab_summary:
        st.markdown("### 📄 Contradiction-Aware Synthesis Report")
        st.download_button(
            "📥 Download Full Summary (.md)",
            summary_markdown,
            "final_summary.md",
            "text/markdown",
            use_container_width=True
        )
        st.markdown(summary_markdown)

    # ================= TAB 5: SOURCE ARTICLES =================
    with tab_articles:
        st.markdown("### 📰 Input News Articles")
        st.caption("Raw source documents ingested into the current analysis session.")

        source_files = sorted(glob.glob("data/raw_sources/*.txt") + glob.glob("data/custom_sources/*.txt"))
        source_files = [f for f in source_files if not os.path.basename(f).startswith("README")]

        if source_files:
            art_tabs = st.tabs([os.path.basename(f) for f in source_files])
            for idx, fp in enumerate(source_files):
                with art_tabs[idx]:
                    with open(fp, "r", encoding="utf-8") as sf:
                        text = sf.read()
                    st.code(text, language="text")
        else:
            st.warning("No source files found in data/raw_sources/ or data/custom_sources/.")

    # ================= TAB 6: HOW IT WORKS =================
    with tab_architecture:
        st.markdown("### 🧠 How the System Works (Viva Defense Architecture)")
        st.markdown("""
        Our 5-stage modular pipeline transforms conflicting news text into verified intelligence:
        """)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.info("**1. Claim Extraction**\n\nGroq LLM breaks compound sentences into verifiable atomic claims with attribution & normalized counts.")
        with c2:
            st.info("**2. Semantic Clustering**\n\n`all-MiniLM-L6-v2` embeds claims; Agglomerative Clustering groups them by topic using adaptive silhouette distance.")
        with c3:
            st.info("**3. Neuro-Symbolic Stance**\n\nDeterministic rules check numbers/dates; LLM CoT evaluates nuances into 5 classes (Agree, Disagree, Discuss, Unrelated, Insufficient).")
        with c4:
            st.info("**4. Contradiction Synthesis**\n\nProduces side-by-side juxtaposition of conflicting perspectives with 100% source attribution.")

        st.markdown("---")
        st.subheader("Data Flow Pipeline")
        st.code("""
[Raw News Articles] ──► [LLM Claim Extraction] ──► [output/claims.json]
                              │
                              ▼
                [Sentence-Transformers MiniLM]
                              │
                              ▼
                [Agglomerative Clustering] ──► [Same-Fact Topic Groups]
                              │
                              ▼
             [Neuro-Symbolic Stance Engine] ──► [output/clusters_with_stance.json]
                              │
                              ▼
             [Contradiction-Aware Synthesis] ──► [output/final_summary.md]
        """, language="text")

    # ================= TAB 7: RUN / LIVE PIPELINE =================
    with tab_run:
        st.markdown("### 🚀 Trigger Live Analysis Pipeline")
        st.caption("You can run the full AI analysis live on default corpus, upload files, or paste custom breaking news text.")

        input_choice = st.radio(
            "Select Input Mode:",
            ["Use Built-in Assam Floods Articles (4 Outlets)", "Upload .txt News Files", "Paste Custom News Text Directly"],
            index=0
        )

        custom_target_paths = None

        if input_choice == "Upload .txt News Files":
            uploaded_files = st.file_uploader(
                "Upload 2 or more conflicting news .txt files",
                type=["txt"],
                accept_multiple_files=True
            )
            if uploaded_files:
                os.makedirs(CUSTOM_SOURCES_DIR, exist_ok=True)
                for f in glob.glob(f"{CUSTOM_SOURCES_DIR}/*"):
                    try:
                        os.remove(f)
                    except Exception:
                        pass
                custom_target_paths = []
                for uf in uploaded_files:
                    fp = os.path.join(CUSTOM_SOURCES_DIR, uf.name)
                    with open(fp, "wb") as f_out:
                        f_out.write(uf.getbuffer())
                    custom_target_paths.append(fp)
                st.success(f"Loaded {len(custom_target_paths)} files ready for processing!")

        elif input_choice == "Paste Custom News Text Directly":
            st.markdown("#### Paste Breaking News Reports (Side-by-Side Analysis)")
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                src_name_1 = st.text_input("Outlet 1 Name:", value="Dainik Jagran", key="p_src1")
                text_1 = st.text_area("Article 1 Text:", height=180, key="p_txt1", placeholder="Paste first article text...")
            with col_p2:
                src_name_2 = st.text_input("Outlet 2 Name:", value="Hindustan Times", key="p_src2")
                text_2 = st.text_area("Article 2 Text:", height=180, key="p_txt2", placeholder="Paste second conflicting article text...")

            if text_1.strip() and text_2.strip():
                os.makedirs(CUSTOM_SOURCES_DIR, exist_ok=True)
                for f in glob.glob(f"{CUSTOM_SOURCES_DIR}/*"):
                    try:
                        os.remove(f)
                    except Exception:
                        pass
                f1_path = os.path.join(CUSTOM_SOURCES_DIR, f"{src_name_1.replace(' ', '_').lower()}.txt")
                f2_path = os.path.join(CUSTOM_SOURCES_DIR, f"{src_name_2.replace(' ', '_').lower()}.txt")
                with open(f1_path, "w", encoding="utf-8") as f:
                    f.write(text_1)
                with open(f2_path, "w", encoding="utf-8") as f:
                    f.write(text_2)
                custom_target_paths = [f1_path, f2_path]
                st.success("Custom news reports loaded into workspace!")

        col_cfg1, col_cfg2 = st.columns(2)
        with col_cfg1:
            auto_thresh = st.checkbox("Adaptive Silhouette Threshold (Recommended)", value=True)
            custom_thresh = None
            if not auto_thresh:
                custom_thresh = st.slider("Manual Distance Threshold", 0.30, 0.80, 0.55, 0.05)

        run_btn = st.button("⚡ Run Full AI Pipeline Now", type="primary", use_container_width=True)

        if run_btn:
            if not get_groq_api_key():
                st.error("🔑 Groq API Key is required! Please provide it in the left sidebar or .env file.")
                st.stop()

            if input_choice != "Use Built-in Assam Floods Articles (4 Outlets)" and not custom_target_paths:
                st.error("Please provide at least 2 source articles to analyze!")
                st.stop()

            with st.spinner("Executing end-to-end contradiction analysis pipeline..."):
                prog = st.progress(0)
                status = st.empty()

                status.info("Stage 1/3: Extracting atomic claims via Groq LLM...")
                prog.progress(25)
                run_extraction_pipeline(input_files=custom_target_paths)

                status.info("Stage 2/3: Semantic Clustering & Neuro-Symbolic Stance Engine...")
                prog.progress(65)
                run_stance_pipeline(distance_threshold=custom_thresh)

                status.info("Stage 3/3: Synthesizing Contradiction-Aware Summary Report...")
                prog.progress(90)
                run_summary_pipeline()

                prog.progress(100)
                status.success("✅ Analysis completed successfully! Refreshing dashboard...")
                st.rerun()

else:
    st.info("No prior analysis results found. Click below or open Tab 7 to run the pipeline.")
    if st.button("🚀 Run Pipeline on Default News Corpus Now", type="primary"):
        with st.spinner("Processing..."):
            run_extraction_pipeline()
            run_stance_pipeline()
            run_summary_pipeline()
            st.rerun()
