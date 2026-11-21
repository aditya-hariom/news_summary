"""
Contradiction-Aware News Analysis System — Student & Viva Friendly Dashboard
----------------------------------------------------------------------------
A clean, minimal, beginner-friendly interface designed for a 5-minute college viva demonstration.
Visualizes news contradictions side-by-side with 100% source attribution.
"""

import os
import sys
import json
import glob
import streamlit as st
from dotenv import load_dotenv

# Ensure project root is in sys.path
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
from src.extraction import run_extraction_pipeline
from src.stance_classifier import run_stance_pipeline
from src.summarizer import run_summary_pipeline

# 1. Page Configuration
st.set_page_config(
    page_title="News Contradiction Analyzer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Clean, Minimal, Consistent Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Clean Top Header */
    .app-header {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        color: #F8FAFC;
        padding: 1.8rem 2rem;
        border-radius: 14px;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
    }
    .app-title {
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(90deg, #38BDF8, #818CF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .app-subtitle {
        color: #94A3B8;
        font-size: 1rem;
        margin-top: 0.4rem;
        margin-bottom: 0;
    }

    /* Status & Badge Pills */
    .pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        margin-right: 6px;
    }
    .pill-agree { background: #DCFCE7; color: #15803D; }
    .pill-conflict { background: #FFE4E6; color: #BE123C; }
    .pill-discuss { background: #FEF3C7; color: #B45309; }
    .pill-unrelated { background: #F1F5F9; color: #475569; }
    .pill-insufficient { background: #F3E8FF; color: #6B21A8; }

    /* Contradiction VS Card */
    .conflict-card {
        background: #FFFFFF;
        border: 2px solid #FDA4AF;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 2px 10px rgba(244, 63, 94, 0.06);
    }
    .source-box-a {
        background: #FFF1F2;
        border: 1px solid #FECDD3;
        border-radius: 8px;
        padding: 0.9rem;
        height: 100%;
    }
    .source-box-b {
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 8px;
        padding: 0.9rem;
        height: 100%;
    }
    .vs-text {
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 1.3rem;
        color: #E11D48;
    }

    /* Agreed Fact Card */
    .agree-card {
        background: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-left: 5px solid #16A34A;
        border-radius: 8px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.8rem;
    }

    /* Claims Card */
    .claim-item {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.9rem;
        margin-bottom: 0.7rem;
    }

    /* How it works card */
    .step-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem;
        height: 100%;
    }
</style>
""", unsafe_allow_html=True)

# 3. Clean Header
st.markdown("""
<div class="app-header">
    <h1 class="app-title">⚖️ Contradiction-Aware News Analysis</h1>
    <p class="app-subtitle">
        An AI system that compares news reports on breaking events, finds agreements and contradictions, and presents them side-by-side with source attribution.
    </p>
</div>
""", unsafe_allow_html=True)

# 4. Sidebar: Simple Controls & API Status
with st.sidebar:
    st.markdown("### ⚙️ System Setup")
    api_key_status = get_groq_api_key()
    if api_key_status:
        st.success("🟢 **Groq API**: Connected")
    else:
        st.warning("🟡 **Groq API Key Required**")
        user_key = st.text_input(
            "Enter Groq API Key:",
            type="password",
            placeholder="gsk_...",
            help="Required to run analysis on new articles. Get a free key at console.groq.com"
        )
        if user_key:
            os.environ["GROQ_API_KEY"] = user_key.strip()
            st.success("API key registered!")
            st.rerun()

    st.markdown("---")
    st.markdown("### 💡 Quick Viva Guide")
    st.markdown("**What makes this unique?**")
    st.caption("Standard AI summarizers blend conflicting numbers into an average. Our system highlights disputes side-by-side without guessing.")
    
    st.markdown("---")
    st.caption("College Minor Project • Aditya")

# 5. Load Data & Prepare Metrics
has_data = os.path.exists(CLUSTERS_FILE) and os.path.exists(CLAIMS_FILE) and os.path.exists(SUMMARY_FILE)

claims_list = []
clusters_dict = {}
summary_md = ""

if has_data:
    try:
        with open(CLAIMS_FILE, "r", encoding="utf-8") as f:
            claims_list = json.load(f)
        with open(CLUSTERS_FILE, "r", encoding="utf-8") as f:
            clusters_dict = json.load(f)
        with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
            summary_md = f.read()
    except Exception:
        has_data = False

# Organize clusters for display
disputed_list = []
agree_list = []
developing_list = []
single_list = []

for cid, cdata in clusters_dict.items():
    claims = cdata.get("claims", [])
    edges = cdata.get("edges", [])
    sources = set(c.get("source_name") or c.get("source_file") for c in claims)
    
    has_disagree = any(e.get("stance") == "DISAGREE" for e in edges)
    has_agree = any(e.get("stance") == "AGREE" for e in edges)
    has_discuss = any(e.get("stance") == "DISCUSS" for e in edges)

    if len(sources) <= 1:
        single_list.append(cdata)
    elif has_disagree:
        disputed_list.append(cdata)
    elif has_agree:
        agree_list.append(cdata)
    elif has_discuss:
        developing_list.append(cdata)
    else:
        single_list.append(cdata)

# Count total unique sources
all_sources = sorted(set(c.get("source_name") or c.get("source_file") for c in claims_list))

# 6. Main Five Simple Tabs
tab_home, tab_analyze, tab_results, tab_explorer, tab_about = st.tabs([
    "🏠 Home / Overview",
    "📰 Analyze News",
    "📊 Results",
    "🔍 Claims Explorer",
    "🧠 About / How It Works",
])

# ==============================================================================
# TAB 1: HOME / OVERVIEW
# ==============================================================================
with tab_home:
    st.markdown("### 📌 Welcome to Contradiction-Aware News Analysis")
    st.write(
        "When breaking news happens, different newspapers often report conflicting death tolls, dates, or casualty numbers. "
        "Standard AI tools (like ChatGPT) often blend these numbers together into a single misleading average. "
        "**This project solves that problem by keeping facts separate and displaying contradictions side-by-side.**"
    )

    # 4 Simple Steps of the Pipeline
    st.markdown("#### 🔄 How Information Flows Through Our System")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="step-card">
            <h4 style="margin:0 0 6px 0; color:#1E40AF;">1. Extract Facts</h4>
            <p style="font-size:0.85rem; color:#475569; margin:0;">
                The LLM reads news articles and breaks compound sentences into atomic, verifiable facts.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="step-card">
            <h4 style="margin:0 0 6px 0; color:#5B21B6;">2. Group by Topic</h4>
            <p style="font-size:0.85rem; color:#475569; margin:0;">
                AI sentence embeddings group claims that discuss the exact same factual topic together.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="step-card">
            <h4 style="margin:0 0 6px 0; color:#BE123C;">3. Detect Conflicts</h4>
            <p style="font-size:0.85rem; color:#475569; margin:0;">
                Rules and AI compare statements to check if sources agree, contradict, or report updates.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="step-card">
            <h4 style="margin:0 0 6px 0; color:#065F46;">4. Clear Summary</h4>
            <p style="font-size:0.85rem; color:#475569; margin:0;">
                Generates a clean report showing agreements, conflicts, and single-source claims side-by-side.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Current Project Status Card
    st.markdown("#### 📈 Current Active News Analysis Status")
    if has_data:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Articles Analyzed", f"{len(all_sources)} Outlets", help="Independent news organizations in this batch")
        with m2:
            st.metric("Facts Extracted", f"{len(claims_list)} Claims", help="Individual verifiable statements extracted")
        with m3:
            st.metric("Conflicts Found", f"{len(disputed_list)} Disputed Topics", help="Factual contradictions identified between outlets")
        with m4:
            st.metric("Agreed Facts", f"{len(agree_list)} Topics", help="Facts confirmed by multiple outlets")

        st.info("💡 **Ready to explore?** Click on the **📊 Results** tab above to see the detected conflicts side-by-side, or go to **📰 Analyze News** to test your own articles.")
    else:
        st.warning("No analysis results found yet. Go to the **📰 Analyze News** tab to run your first analysis!")

# ==============================================================================
# TAB 2: ANALYZE NEWS
# ==============================================================================
with tab_analyze:
    st.markdown("### 📰 Run News Analysis")
    st.caption("Choose how you want to input articles: paste text directly, upload .txt files, or use the built-in disaster corpus.")

    input_mode = st.radio(
        "Choose Input Source:",
        [
            "⚡ Paste News Text Directly (Quickest for Viva Demo)",
            "📁 Upload .txt Files (Multiple Articles)",
            "📚 Use Built-in 2026 Assam Floods News Corpus"
        ],
        index=0
    )

    custom_files = None

    if input_mode == "⚡ Paste News Text Directly (Quickest for Viva Demo)":
        st.markdown("##### Paste Two Conflicting News Reports")
        
        # Quick Demo Button to pre-fill conflicting examples
        if st.button("📋 Load Sample Conflicting Story (Assam Floods Casualties)", type="secondary"):
            st.session_state["p_src_1"] = "Times of India"
            st.session_state["p_txt_1"] = "On 2 August 2026, state officials reported that 82 people have died across Assam due to severe flooding. Over 1.78 lakh people have been affected."
            st.session_state["p_src_2"] = "MapsOfIndia Wire"
            st.session_state["p_txt_2"] = "According to official reports, the death toll in the 2026 Assam floods has swelled to 47. More than 7.2 lakh people have been affected."

        col_a, col_b = st.columns(2)
        with col_a:
            src1 = st.text_input("News Outlet 1 Name:", value=st.session_state.get("p_src_1", "Times of India"), key="src1_in")
            txt1 = st.text_area("Article 1 Text:", value=st.session_state.get("p_txt_1", ""), height=150, placeholder="Paste first news story here...", key="txt1_in")
        with col_b:
            src2 = st.text_input("News Outlet 2 Name:", value=st.session_state.get("p_src_2", "MapsOfIndia Wire"), key="src2_in")
            txt2 = st.text_area("Article 2 Text:", value=st.session_state.get("p_txt_2", ""), height=150, placeholder="Paste second news story here...", key="txt2_in")

        if txt1.strip() and txt2.strip():
            os.makedirs(CUSTOM_SOURCES_DIR, exist_ok=True)
            for f in glob.glob(f"{CUSTOM_SOURCES_DIR}/*"):
                try: os.remove(f)
                except Exception: pass
            p1 = os.path.join(CUSTOM_SOURCES_DIR, f"{src1.replace(' ', '_').lower()}.txt")
            p2 = os.path.join(CUSTOM_SOURCES_DIR, f"{src2.replace(' ', '_').lower()}.txt")
            with open(p1, "w", encoding="utf-8") as f: f.write(txt1)
            with open(p2, "w", encoding="utf-8") as f: f.write(txt2)
            custom_files = [p1, p2]

    elif input_mode == "📁 Upload .txt Files (Multiple Articles)":
        uploaded = st.file_uploader(
            "Upload 2 or more news articles as .txt files:",
            type=["txt"],
            accept_multiple_files=True
        )
        if uploaded:
            os.makedirs(CUSTOM_SOURCES_DIR, exist_ok=True)
            for f in glob.glob(f"{CUSTOM_SOURCES_DIR}/*"):
                try: os.remove(f)
                except Exception: pass
            custom_files = []
            for u in uploaded:
                p = os.path.join(CUSTOM_SOURCES_DIR, u.name)
                with open(p, "wb") as f: f.write(u.getbuffer())
                custom_files.append(p)
            st.success(f"Loaded {len(custom_files)} files ready for analysis!")

    st.markdown("---")
    run_button = st.button("🚀 Run Analysis Pipeline", type="primary", use_container_width=True)

    if run_button:
        if not get_groq_api_key():
            st.error("🔑 Groq API Key is missing! Please enter it in the left sidebar.")
            st.stop()

        if input_mode != "📚 Use Built-in 2026 Assam Floods News Corpus" and not custom_files:
            st.error("Please provide at least 2 news articles to compare!")
            st.stop()

        with st.spinner("Analyzing articles... (Extracting facts, clustering, and finding conflicts)"):
            prog = st.progress(0)
            
            prog.progress(25)
            run_extraction_pipeline(input_files=custom_files)

            prog.progress(65)
            run_stance_pipeline()

            prog.progress(90)
            run_summary_pipeline()

            prog.progress(100)
            st.success("✅ Analysis Complete! Switch to the **Results** tab to view findings.")
            st.rerun()

# ==============================================================================
# TAB 3: RESULTS
# ==============================================================================
with tab_results:
    st.markdown("### 📊 Contradiction-Aware Analysis Results")
    st.caption("Here is what our AI found after cross-checking all articles.")

    if not has_data:
        st.info("No results to display. Go to the **Analyze News** tab to run your first test!")
    else:
        # Simple Human-Readable Legend
        st.markdown("""
        <div>
            <span class="pill pill-conflict">🔴 Reports Conflict</span>
            <span class="pill pill-agree">🟢 Reports Agree</span>
            <span class="pill pill-discuss">🟡 Different Time / Context</span>
            <span class="pill pill-unrelated">⚪ Single Source</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")

        # 1. DISPUTED FACTS SECTION
        st.markdown("#### 🚨 Conflicting Information (Placed Side-by-Side)")
        st.write("These statements directly contradict each other. The system does not pick a winner; it highlights both with their sources.")

        if not disputed_list:
            st.success("🎉 No factual contradictions detected! All analyzed sources agree or report complementary details.")
        else:
            for idx, cluster in enumerate(disputed_list, 1):
                topic = cluster.get("topic", "Incident").replace("_", " ").title()
                disagree_edges = [e for e in cluster.get("edges", []) if e.get("stance") == "DISAGREE"]

                seen = set()
                for e in disagree_edges:
                    key = tuple(sorted([e["claim_a"], e["claim_b"]]))
                    if key in seen:
                        continue
                    seen.add(key)

                    attr = e.get("conflicting_attribute") or topic
                    reasoning = e.get("reasoning", "Discrepancy detected.")

                    st.markdown(f"""
                    <div class="conflict-card">
                        <div style="font-weight:700; color:#BE123C; font-size:1.05rem; margin-bottom:10px;">
                            ⚡ Conflict #{idx}: {attr} Discrepancy
                        </div>
                        <div style="display:flex; gap:12px; align-items:stretch;">
                            <div style="flex:5;" class="source-box-a">
                                <span style="font-size:0.75rem; font-weight:700; background:#FECDD3; color:#9F1239; padding:2px 8px; border-radius:4px;">
                                    {e['source_a']} {f"({e['date_a']})" if e.get('date_a') else ''}
                                </span>
                                <h4 style="color:#991B1B; margin:8px 0 0 0; font-size:1.05rem;">"{e['claim_a']}"</h4>
                            </div>
                            <div style="flex:1;" class="vs-text">VS</div>
                            <div style="flex:5;" class="source-box-b">
                                <span style="font-size:0.75rem; font-weight:700; background:#BFDBFE; color:#1E40AF; padding:2px 8px; border-radius:4px;">
                                    {e['source_b']} {f"({e['date_b']})" if e.get('date_b') else ''}
                                </span>
                                <h4 style="color:#1D4ED8; margin:8px 0 0 0; font-size:1.05rem;">"{e['claim_b']}"</h4>
                            </div>
                        </div>
                        <div style="margin-top:10px; font-size:0.85rem; color:#881337; background:#FFF1F2; padding:8px 12px; border-radius:6px;">
                            <b>🔍 Why this is a conflict:</b> {reasoning}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("---")

        # 2. AGREED FACTS SECTION
        st.markdown("#### ✅ Consistently Reported Facts (Agreement)")
        st.write("These facts were confirmed by two or more independent media outlets.")

        if not agree_list:
            st.info("No multi-source identical agreements found in this batch.")
        else:
            for idx, cluster in enumerate(agree_list, 1):
                sources = sorted(set(m.get("source_name") or m.get("source_file") for m in cluster["claims"]))
                rep_claim = cluster["claims"][0]["claim"]
                st.markdown(f"""
                <div class="agree-card">
                    <div style="font-weight:700; font-size:1rem; color:#14532D; margin-bottom:4px;">
                        🟢 Fact #{idx}: "{rep_claim}"
                    </div>
                    <div style="font-size:0.82rem; color:#166534;">
                        <b>Reported by:</b> {' • '.join([f'<code>{s}</code>' for s in sources])}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # 3. SINGLE SOURCE & DEVELOPING CLAIMS
        with st.expander(f"ℹ️ View Single-Source & Developing Claims ({len(single_list)} items)", expanded=False):
            st.caption("These statements appeared in only ONE outlet or reflect evolving dates. They are not disputed, but remain uncorroborated.")
            for c in single_list:
                for m in c["claims"]:
                    src = m.get("source_name") or m.get("source_file")
                    d_str = f" `[{m['date']}]`" if m.get("date") else ""
                    st.markdown(f"- **[{src}]**{d_str}: \"{m['claim']}\"")

        # 4. DOWNLOAD FULL REPORT
        st.markdown("---")
        st.download_button(
            "📥 Download Complete Summary Report (.md)",
            summary_md,
            "news_analysis_report.md",
            "text/markdown",
            use_container_width=True
        )

# ==============================================================================
# TAB 4: CLAIMS EXPLORER
# ==============================================================================
with tab_explorer:
    st.markdown("### 🔍 Claims Explorer")
    st.caption("Search, filter, and inspect individual factual statements extracted by the AI.")

    if not claims_list:
        st.info("No extracted claims found. Run an analysis first!")
    else:
        col_s, col_src, col_top = st.columns([2, 1, 1])
        with col_s:
            q = st.text_input("🔎 Search by word:", placeholder="e.g. death, affected, water, rescue", help="Filters claims containing this keyword")
        with col_src:
            selected_src = st.selectbox("Filter by Outlet:", ["All Outlets"] + all_sources)
        with col_top:
            all_topics = sorted(set(c.get("category") or c.get("topic", "general") for c in claims_list))
            selected_top = st.selectbox("Filter by Topic:", ["All Topics"] + all_topics)

        filtered = claims_list
        if q:
            filtered = [c for c in filtered if q.lower() in c.get("claim", "").lower()]
        if selected_src != "All Outlets":
            filtered = [c for c in filtered if (c.get("source_name") or c.get("source_file")) == selected_src]
        if selected_top != "All Topics":
            filtered = [c for c in filtered if (c.get("category") or c.get("topic")) == selected_top]

        st.info(f"Showing **{len(filtered)}** claims:")

        for c in filtered:
            cid = c.get("claim_id", "N/A")
            src = c.get("source_name") or c.get("source_file")
            topic = c.get("category") or c.get("topic", "general")
            val = f" | Number: `{c['numerical_value']} {c.get('numerical_unit', '')}`" if c.get("numerical_value") is not None else ""
            dt = f" | Date: `{c['date']}`" if c.get("date") else ""
            attr = f" | Attribution: *{c['attribution']}*" if c.get("attribution") else ""

            st.markdown(f"""
            <div class="claim-item">
                <div style="font-size:0.75rem; color:#64748B; margin-bottom:4px;">
                    <code>{cid}</code> • <b>Outlet:</b> <code>{src}</code> • <b>Topic:</b> <code>{topic}</code>
                    {val}{dt}{attr}
                </div>
                <div style="font-size:0.95rem; font-weight:600; color:#0F172A;">
                    "{c.get('claim')}"
                </div>
            </div>
            """, unsafe_allow_html=True)

# ==============================================================================
# TAB 5: ABOUT / HOW IT WORKS
# ==============================================================================
with tab_about:
    st.markdown("### 🧠 How the System Works (Viva Explanation Guide)")
    st.write(
        "This project is a **College Minor Project** designed to demonstrate how NLP and Large Language Models "
        "can detect conflicting information across news sources."
    )

    st.markdown("#### 📖 Key Definitions for Your Viva")
    v1, v2 = st.columns(2)
    with v1:
        st.markdown("""
        - **Atomic Claim:** A single, minimal factual statement that can be verified independently (e.g. *"5 people died"*). Breaking news paragraphs into atomic claims prevents confusion.
        - **Semantic Clustering:** Using sentence embeddings (`all-MiniLM-L6-v2`) to group claims talking about the same topic, even if they use different words.
        """)
    with v2:
        st.markdown("""
        - **Neuro-Symbolic Stance:** Combining mathematical rules (for exact numbers and dates) with an LLM (for linguistic intent) to classify relationships accurately.
        - **Contradiction-Awareness:** Presenting conflicting numbers side-by-side with source names, rather than hiding disagreements in an artificial summary.
        """)

    st.markdown("---")
    st.markdown("#### 🏷️ Meaning of Relationship Labels")
    st.markdown("""
    - 🟢 **Reports Agree (`AGREE`)**: Both news outlets report the exact same fact or number.
    - 🔴 **Reports Conflict (`DISAGREE`)**: Outlets report incompatible figures or outcomes for the same event window.
    - 🟡 **May Describe Different Time / Context (`DISCUSS`)**: Numbers differ because one report was published earlier and one later as the situation evolved.
    - ⚪ **Claims About Different Things (`UNRELATED`)**: Statements belong to completely different topics (e.g. food packets vs death tolls).
    - ⚠️ **Not Enough Information to Decide (`INSUFFICIENT_EVIDENCE`)**: Vague wording without dates or numbers to make a confident comparison.
    """)

    st.markdown("---")
    with st.expander("📰 View Raw Source Articles Used in Analysis", expanded=False):
        raw_files = sorted(glob.glob("data/raw_sources/*.txt") + glob.glob("data/custom_sources/*.txt"))
        raw_files = [f for f in raw_files if not os.path.basename(f).startswith("README")]
        if raw_files:
            file_tabs = st.tabs([os.path.basename(f) for f in raw_files])
            for i, fp in enumerate(raw_files):
                with file_tabs[i]:
                    with open(fp, "r", encoding="utf-8") as rf:
                        st.code(rf.read(), language="text")
        else:
            st.write("No source files available.")
