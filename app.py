"""
Contradiction-Aware News Analysis System — Guided Streamlit Dashboard
---------------------------------------------------------------------
A clean, two-view interface for comparing news reports side-by-side,
highlighting direct contradictions, corroborated facts, and timeline updates
with full source attribution.
"""

import glob
import html
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    CLAIMS_FILE,
    CLUSTERS_FILE,
    CUSTOM_SOURCES_DIR,
    EMBEDDING_MODEL,
    GROQ_MODEL,
    RAW_SOURCES_DIR,
    SUMMARY_FILE,
    get_groq_api_key,
)
from src.extraction import run_extraction_pipeline
from src.stance_classifier import run_stance_pipeline
from src.summarizer import categorize_clusters, run_summary_pipeline


st.set_page_config(
    page_title="News Contradiction Analyzer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .app-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        color: #F8FAFC;
        padding: 1.4rem 1.8rem;
        border-radius: 14px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.12);
    }
    .app-title {
        font-size: 1.75rem;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(90deg, #38BDF8, #818CF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .app-subtitle {
        color: #CBD5E1;
        font-size: 0.95rem;
        margin: 0.35rem 0 0 0;
    }

    .verdict-banner {
        padding: 0.9rem 1.15rem;
        border-radius: 10px;
        margin: 0.6rem 0 1rem 0;
        font-size: 0.95rem;
        font-weight: 600;
    }
    .verdict-conflict {
        background: #FFF1F2;
        border: 1px solid #FECDD3;
        border-left: 5px solid #E11D48;
        color: #881337;
    }
    .verdict-clean {
        background: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-left: 5px solid #16A34A;
        color: #14532D;
    }

    .conflict-card {
        background: #FFFFFF;
        border: 1.5px solid #FDA4AF;
        border-radius: 12px;
        padding: 1.1rem 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(244, 63, 94, 0.06);
    }
    .conflict-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 0.75rem;
    }
    .conflict-title {
        font-weight: 700;
        color: #BE123C;
        font-size: 1.02rem;
    }
    .attr-badge {
        font-size: 0.75rem;
        font-weight: 700;
        background: #FFE4E6;
        color: #9F1239;
        padding: 3px 10px;
        border-radius: 9999px;
    }
    .vs-grid {
        display: flex;
        gap: 12px;
        align-items: stretch;
    }
    .source-box-a {
        flex: 5;
        background: #FFF1F2;
        border: 1px solid #FECDD3;
        border-radius: 8px;
        padding: 0.85rem;
    }
    .source-box-b {
        flex: 5;
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 8px;
        padding: 0.85rem;
    }
    .vs-text {
        flex: 0.7;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 1.15rem;
        color: #E11D48;
    }
    .reason-box {
        margin-top: 0.75rem;
        font-size: 0.86rem;
        color: #881337;
        background: #FFF1F2;
        padding: 0.6rem 0.85rem;
        border-radius: 6px;
    }

    .agree-card {
        background: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-left: 5px solid #16A34A;
        border-radius: 8px;
        padding: 0.9rem 1.05rem;
        margin-bottom: 0.75rem;
    }
    .update-card {
        background: #FFFBEB;
        border: 1px solid #FDE68A;
        border-left: 5px solid #D97706;
        border-radius: 8px;
        padding: 0.9rem 1.05rem;
        margin-bottom: 0.75rem;
    }
    .claim-item {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.65rem;
    }
    .step-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 0.95rem;
        height: 100%;
    }
</style>
"""


def _esc(value: Any) -> str:
    """Safely HTML-escape text rendered inside markdown HTML blocks."""
    return html.escape("" if value is None else str(value), quote=True)


def _safe_filename(name: str, fallback: str) -> str:
    """Convert an outlet name into a safe lowercase .txt filename."""
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip().lower()).strip("_")
    return f"{cleaned or fallback}.txt"


def _clear_custom_sources() -> None:
    """Remove temporary custom source files while keeping .gitkeep intact."""
    os.makedirs(CUSTOM_SOURCES_DIR, exist_ok=True)
    for path in glob.glob(os.path.join(CUSTOM_SOURCES_DIR, "*")):
        if os.path.basename(path) == ".gitkeep":
            continue
        try:
            if os.path.isfile(path):
                os.remove(path)
        except OSError:
            pass


def _clear_saved_outputs() -> None:
    """Delete previous pipeline output files so the user starts fresh."""
    for path in (CLAIMS_FILE, CLUSTERS_FILE, SUMMARY_FILE):
        try:
            if os.path.isfile(path):
                os.remove(path)
        except OSError:
            pass


def _load_sample_into_state() -> None:
    """Populate the paste boxes with a built-in conflicting 2-article example."""
    st.session_state["p_src_1"] = "Times of India"
    st.session_state["p_txt_1"] = (
        "State officials confirmed that 82 people have died across Assam due to "
        "severe flooding. Over 1.78 lakh people have been affected."
    )
    st.session_state["p_src_2"] = "MapsOfIndia Wire"
    st.session_state["p_txt_2"] = (
        "According to official disaster reports, the death toll in the Assam floods "
        "has reached 47. More than 7.2 lakh people have been affected."
    )
    st.session_state["input_mode_radio"] = "Paste 2 Articles"
    st.session_state["expand_input_panel"] = True


def _load_analysis_outputs() -> Tuple[bool, List[Dict[str, Any]], Dict[str, Any], str]:
    """Load generated claims, clusters, and markdown summary if present."""
    required = (CLAIMS_FILE, CLUSTERS_FILE, SUMMARY_FILE)
    if not all(os.path.exists(p) for p in required):
        return False, [], {}, ""

    try:
        with open(CLAIMS_FILE, "r", encoding="utf-8") as f:
            claims_list = json.load(f)
        with open(CLUSTERS_FILE, "r", encoding="utf-8") as f:
            clusters_dict = json.load(f)
        with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
            summary_md = f.read()
        return True, claims_list, clusters_dict, summary_md
    except Exception:
        return False, [], {}, ""


def _extract_unique_disagree_pairs(disputed_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collect deduplicated DISAGREE edges across disputed clusters."""
    pairs: List[Dict[str, Any]] = []
    seen = set()
    for cluster in disputed_list:
        topic = cluster.get("topic", "Incident").replace("_", " ").title()
        for edge in cluster.get("edges", []):
            if edge.get("stance") != "DISAGREE":
                continue
            key = tuple(sorted([edge.get("claim_a", ""), edge.get("claim_b", "")]))
            if key in seen:
                continue
            seen.add(key)
            pairs.append({**edge, "_topic": topic})
    return pairs


def _get_builtin_corpus_files() -> List[str]:
    """Return built-in news corpus files excluding documentation READMEs."""
    candidates = sorted(glob.glob(os.path.join(RAW_SOURCES_DIR, "*.txt")))
    return [
        f for f in candidates
        if os.path.isfile(f) and not os.path.basename(f).startswith("README")
    ]


def _run_pipeline(target_files: Optional[List[str]]) -> None:
    """Execute the 3-stage NLP pipeline with live progress updates."""
    if not get_groq_api_key():
        st.error("Groq API Key is missing. Please add it in the left sidebar first.")
        return

    try:
        with st.status("Running Contradiction-Aware Analysis...", expanded=True) as status:
            st.write("1/3 Extracting atomic factual claims from articles...")
            claims = run_extraction_pipeline(input_files=target_files)
            if not claims:
                status.update(label="No claims extracted", state="error")
                st.error("No factual claims could be extracted from the provided input.")
                return

            st.write("2/3 Grouping claims by topic and checking cross-source stances...")
            run_stance_pipeline()

            st.write("3/3 Building side-by-side summary report...")
            run_summary_pipeline()

            status.update(label="Analysis complete!", state="complete", expanded=False)

        st.session_state["expand_input_panel"] = False
        st.rerun()
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")


def _render_sidebar(has_data: bool) -> None:
    """Render sidebar setup, quick sample loaders, and reset controls."""
    with st.sidebar:
        st.markdown("### ⚙️ Setup & Shortcuts")
        if get_groq_api_key():
            st.success("🟢 **Groq API**: Connected")
        else:
            st.warning("🟡 **Groq API Key Required**")
            user_key = st.text_input(
                "Enter Groq API Key:",
                type="password",
                placeholder="gsk_...",
                help="Required to run analysis on articles. Get a key at console.groq.com",
            )
            if user_key:
                os.environ["GROQ_API_KEY"] = user_key.strip()
                st.success("API key saved for this session.")
                st.rerun()

        st.markdown("---")
        st.markdown("### ⚡ Quick Demo Presets")
        if st.button("📋 Load 2-Outlet Flood Conflict", use_container_width=True):
            _load_sample_into_state()
            st.rerun()

        if st.button("📚 Select Built-in 4-Source Corpus", use_container_width=True):
            st.session_state["input_mode_radio"] = "Built-in 2026 Assam Corpus"
            st.session_state["expand_input_panel"] = True
            st.rerun()

        if has_data:
            st.markdown("---")
            if st.button("🗑️ Clear Current Results", use_container_width=True):
                _clear_saved_outputs()
                st.session_state["expand_input_panel"] = True
                st.rerun()

        st.markdown("---")
        st.caption(f"Model: `{GROQ_MODEL}`  \nEmbeddings: `{EMBEDDING_MODEL}`")


def _render_input_panel(has_data: bool) -> None:
    """Render Step 1 (article selection/pasting) inside a clean collapsible panel."""
    expanded = st.session_state.get("expand_input_panel", not has_data)
    panel_title = (
        "📝 Step 1: Compare News Articles (click to edit or run a new batch)"
        if has_data
        else "📝 Step 1: Choose or Paste News Articles to Compare"
    )

    st.session_state.setdefault("input_mode_radio", "Paste 2 Articles")
    st.session_state.setdefault("p_src_1", "Times of India")
    st.session_state.setdefault("p_txt_1", "")
    st.session_state.setdefault("p_src_2", "MapsOfIndia Wire")
    st.session_state.setdefault("p_txt_2", "")

    with st.expander(panel_title, expanded=expanded):
        mode = st.radio(
            "Input Method:",
            ["Paste 2 Articles", "Upload .txt Files", "Built-in 2026 Assam Corpus"],
            horizontal=True,
            key="input_mode_radio",
        )

        target_files: Optional[List[str]] = None
        ready_to_run = True

        if mode == "Paste 2 Articles":
            col_top_l, col_top_r = st.columns([3, 1])
            with col_top_l:
                st.caption("Paste two reports covering the same news event to spot agreements and conflicting figures.")
            with col_top_r:
                if st.button("Load Sample Story", type="secondary", use_container_width=True):
                    _load_sample_into_state()
                    st.rerun()

            col_a, col_b = st.columns(2)
            with col_a:
                src1 = st.text_input("Outlet 1 Name", key="p_src_1")
                txt1 = st.text_area(
                    "Outlet 1 Article Text",
                    height=150,
                    placeholder="Paste first news article here...",
                    key="p_txt_1",
                )
            with col_b:
                src2 = st.text_input("Outlet 2 Name", key="p_src_2")
                txt2 = st.text_area(
                    "Outlet 2 Article Text",
                    height=150,
                    placeholder="Paste second news article here...",
                    key="p_txt_2",
                )

            if not txt1.strip() or not txt2.strip():
                ready_to_run = False

        elif mode == "Upload .txt Files":
            uploaded = st.file_uploader(
                "Upload 2 or more `.txt` news articles:",
                type=["txt"],
                accept_multiple_files=True,
            )
            if not uploaded or len(uploaded) < 2:
                st.info("Upload at least 2 `.txt` files to cross-check them.")
                ready_to_run = False

        else:
            corpus_files = _get_builtin_corpus_files()
            if corpus_files:
                file_names = ", ".join(f"`{os.path.basename(p)}`" for p in corpus_files)
                st.info(f"Ready to analyze **{len(corpus_files)} built-in articles**: {file_names}")
                target_files = corpus_files
            else:
                st.warning("No built-in `.txt` files found in `data/raw_sources/`.")
                ready_to_run = False

        run_clicked = st.button(
            "🚀 Analyze & Compare Articles",
            type="primary",
            use_container_width=True,
            disabled=not ready_to_run,
        )

        if run_clicked:
            if mode == "Paste 2 Articles":
                _clear_custom_sources()
                p1 = os.path.join(CUSTOM_SOURCES_DIR, _safe_filename(src1, "outlet_1"))
                p2 = os.path.join(CUSTOM_SOURCES_DIR, _safe_filename(src2, "outlet_2"))
                if p1 == p2:
                    p2 = os.path.join(CUSTOM_SOURCES_DIR, _safe_filename(f"{src2}_2", "outlet_2"))
                with open(p1, "w", encoding="utf-8") as f:
                    f.write(txt1)
                with open(p2, "w", encoding="utf-8") as f:
                    f.write(txt2)
                target_files = [p1, p2]

            elif mode == "Upload .txt Files":
                _clear_custom_sources()
                target_files = []
                for idx, ufile in enumerate(uploaded, 1):
                    fname = _safe_filename(os.path.splitext(ufile.name)[0], f"upload_{idx}")
                    dest = os.path.join(CUSTOM_SOURCES_DIR, fname)
                    with open(dest, "wb") as f:
                        f.write(ufile.getbuffer())
                    target_files.append(dest)

            _run_pipeline(target_files)


def _render_conflict_cards(disagree_pairs: List[Dict[str, Any]]) -> None:
    """Render side-by-side Outlet A vs Outlet B contradiction cards."""
    if not disagree_pairs:
        st.success("🎉 No direct factual contradictions were detected across the analyzed sources.")
        return

    st.caption("Conflicting claims are kept side-by-side with exact outlet attribution — the system never guesses a winner.")
    for idx, edge in enumerate(disagree_pairs, 1):
        attr = edge.get("conflicting_attribute") or edge.get("_topic", "Fact")
        reasoning = edge.get("reasoning", "Incompatible values or assertions reported.")
        date_a = f" ({_esc(edge['date_a'])})" if edge.get("date_a") else ""
        date_b = f" ({_esc(edge['date_b'])})" if edge.get("date_b") else ""

        st.markdown(
            f"""
            <div class="conflict-card">
                <div class="conflict-header">
                    <span class="conflict-title">⚡ Conflict #{idx}</span>
                    <span class="attr-badge">{_esc(attr)} Discrepancy</span>
                </div>
                <div class="vs-grid">
                    <div class="source-box-a">
                        <span style="font-size:0.76rem; font-weight:700; background:#FECDD3; color:#9F1239; padding:2px 8px; border-radius:4px;">
                            {_esc(edge.get('source_a'))}{date_a}
                        </span>
                        <h4 style="color:#991B1B; margin:8px 0 0 0; font-size:1.02rem; line-height:1.4;">"{_esc(edge.get('claim_a'))}"</h4>
                    </div>
                    <div class="vs-text">VS</div>
                    <div class="source-box-b">
                        <span style="font-size:0.76rem; font-weight:700; background:#BFDBFE; color:#1E40AF; padding:2px 8px; border-radius:4px;">
                            {_esc(edge.get('source_b'))}{date_b}
                        </span>
                        <h4 style="color:#1D4ED8; margin:8px 0 0 0; font-size:1.02rem; line-height:1.4;">"{_esc(edge.get('claim_b'))}"</h4>
                    </div>
                </div>
                <div class="reason-box">
                    <b>Why this conflicts:</b> {_esc(reasoning)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_agreement_cards(agree_list: List[Dict[str, Any]]) -> None:
    """Render corroborated facts confirmed by two or more outlets."""
    if not agree_list:
        st.info("No multi-source exact agreements found in this batch.")
        return

    st.caption("Facts confirmed consistently by 2 or more independent outlets.")
    for idx, cluster in enumerate(agree_list, 1):
        claims = cluster.get("claims", [])
        if not claims:
            continue
        sources = sorted(set(c.get("source_name") or c.get("source_file") or "Unknown" for c in claims))
        rep_claim = claims[0].get("claim", "")
        source_chips = " • ".join(f"<code>{_esc(s)}</code>" for s in sources)

        st.markdown(
            f"""
            <div class="agree-card">
                <div style="font-weight:700; font-size:0.98rem; color:#14532D; margin-bottom:4px;">
                    🟢 Agreed Fact #{idx}: "{_esc(rep_claim)}"
                </div>
                <div style="font-size:0.82rem; color:#166534;">
                    <b>Confirmed by:</b> {source_chips}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_developing_and_single(
    developing_list: List[Dict[str, Any]],
    single_list: List[Dict[str, Any]],
) -> None:
    """Render timeline/context updates (DISCUSS) and single-source claims."""
    st.markdown("##### 🟡 Developing / Timeline Updates (`DISCUSS`)")
    if not developing_list:
        st.info("No timeline or context-evolution updates detected.")
    else:
        st.caption("These claims show different numbers or details because reports come from different dates or scopes.")
        for idx, cluster in enumerate(developing_list, 1):
            topic = cluster.get("topic", "Event Update").replace("_", " ").title()
            discuss_edges = [e for e in cluster.get("edges", []) if e.get("stance") == "DISCUSS"]
            reason = discuss_edges[0].get("reasoning", "") if discuss_edges else ""

            items_html = []
            for c in cluster.get("claims", []):
                src = _esc(c.get("source_name") or c.get("source_file") or "Unknown")
                dt = f" [{_esc(c['date'])}]" if c.get("date") else ""
                items_html.append(f"<li><b>{src}{dt}:</b> \"{_esc(c.get('claim'))}\"</li>")

            reason_html = (
                f"<div style='font-size:0.84rem; color:#92400E; margin-top:6px;'><b>Context:</b> {_esc(reason)}</div>"
                if reason else ""
            )
            st.markdown(
                f"""
                <div class="update-card">
                    <div style="font-weight:700; color:#B45309; margin-bottom:6px;">
                        🕒 Update #{idx}: {_esc(topic)}
                    </div>
                    <ul style="margin:0; padding-left:1.2rem; color:#451A03; font-size:0.92rem;">
                        {''.join(items_html)}
                    </ul>
                    {reason_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("##### ⚪ Single-Source / Uncorroborated Claims")
    if not single_list:
        st.info("No uncorroborated single-source claims in this batch.")
    else:
        st.caption("Reported by only one outlet so far — not disputed, but awaiting independent confirmation.")
        for cluster in single_list:
            for c in cluster.get("claims", []):
                src = c.get("source_name") or c.get("source_file") or "Unknown"
                d_str = f" `[{c['date']}]`" if c.get("date") else ""
                st.markdown(f"- **[{src}]**{d_str}: \"{c.get('claim', '')}\"")


def _render_results_section(
    claims_list: List[Dict[str, Any]],
    clusters_dict: Dict[str, Any],
    summary_md: str,
    all_sources: List[str],
) -> None:
    """Render Step 2: overview verdict, metrics, and categorized result tabs."""
    agree_list, disputed_list, developing_list, single_list = categorize_clusters(clusters_dict)
    disagree_pairs = _extract_unique_disagree_pairs(disputed_list)
    other_count = len(developing_list) + len(single_list)

    st.markdown("### 📊 Step 2: Comparison Results")

    if disagree_pairs:
        st.markdown(
            f"""
            <div class="verdict-banner verdict-conflict">
                ⚠️ Found <b>{len(disagree_pairs)} direct contradiction(s)</b> across {len(all_sources)} outlets.
                Conflicting statements are shown side-by-side below without averaging.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="verdict-banner verdict-clean">
                ✅ No direct contradictions found across {len(all_sources)} outlets ({len(claims_list)} claims analyzed).
            </div>
            """,
            unsafe_allow_html=True,
        )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Outlets Compared", len(all_sources))
    m2.metric("Direct Conflicts", len(disagree_pairs))
    m3.metric("Agreed Facts", len(agree_list))
    m4.metric("Updates & Single-Source", other_count)

    res_tabs = st.tabs([
        f"🔴 Conflicts ({len(disagree_pairs)})",
        f"🟢 Agreements ({len(agree_list)})",
        f"🟡 Updates & Single-Source ({other_count})",
        "📄 Full Report",
    ])

    with res_tabs[0]:
        _render_conflict_cards(disagree_pairs)
    with res_tabs[1]:
        _render_agreement_cards(agree_list)
    with res_tabs[2]:
        _render_developing_and_single(developing_list, single_list)
    with res_tabs[3]:
        st.download_button(
            "📥 Download Complete Report (.md)",
            summary_md,
            file_name="news_analysis_report.md",
            mime="text/markdown",
            use_container_width=True,
        )
        st.markdown("---")
        st.markdown(summary_md)


def _render_explorer_and_guide(claims_list: List[Dict[str, Any]], all_sources: List[str]) -> None:
    """Render Claims Explorer, Viva Cheat-Sheet, and Raw Source Viewer."""
    st.markdown("### 🔍 Claims Explorer")
    if not claims_list:
        st.info("No extracted claims yet. Run an analysis in the **Compare & Results** tab first.")
    else:
        col_q, col_src, col_top = st.columns([2, 1, 1])
        with col_q:
            query = st.text_input("Search claims:", placeholder="e.g. death, affected, Assam, rescue")
        with col_src:
            selected_src = st.selectbox("Filter by Outlet:", ["All Outlets"] + all_sources)
        with col_top:
            all_topics = sorted(set(c.get("category") or c.get("topic", "general") for c in claims_list))
            selected_topic = st.selectbox("Filter by Topic:", ["All Topics"] + all_topics)

        filtered = claims_list
        if query:
            q_low = query.lower()
            filtered = [c for c in filtered if q_low in c.get("claim", "").lower()]
        if selected_src != "All Outlets":
            filtered = [c for c in filtered if (c.get("source_name") or c.get("source_file")) == selected_src]
        if selected_topic != "All Topics":
            filtered = [c for c in filtered if (c.get("category") or c.get("topic")) == selected_topic]

        st.caption(f"Showing **{len(filtered)}** of {len(claims_list)} extracted claims")
        for c in filtered:
            cid = _esc(c.get("claim_id", "N/A"))
            src = _esc(c.get("source_name") or c.get("source_file") or "Unknown")
            topic = _esc(c.get("category") or c.get("topic", "general"))
            val = (
                f" • <b>Value:</b> <code>{_esc(c['numerical_value'])} {_esc(c.get('numerical_unit', ''))}</code>"
                if c.get("numerical_value") is not None else ""
            )
            dt = f" • <b>Date:</b> <code>{_esc(c['date'])}</code>" if c.get("date") else ""
            attr = f" • <b>Attribution:</b> <i>{_esc(c['attribution'])}</i>" if c.get("attribution") else ""

            st.markdown(
                f"""
                <div class="claim-item">
                    <div style="font-size:0.76rem; color:#64748B; margin-bottom:4px;">
                        <code>{cid}</code> • <b>Outlet:</b> <code>{src}</code> • <b>Topic:</b> <code>{topic}</code>{val}{dt}{attr}
                    </div>
                    <div style="font-size:0.95rem; font-weight:600; color:#0F172A;">
                        "{_esc(c.get('claim'))}"
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("### 🧠 How It Works (Viva Quick Guide)")
    c1, c2, c3, c4 = st.columns(4)
    steps_info = [
        ("1. Extract Facts", "#1E40AF", "Groq LLM splits articles into single, verifiable atomic claims with numbers and dates."),
        ("2. Group by Topic", "#5B21B6", "Sentence embeddings (`all-MiniLM-L6-v2`) cluster claims discussing the same topic."),
        ("3. Detect Stances", "#BE123C", "Numeric/date rules + LLM reasoning label pairs as Agree, Conflict, or Update."),
        ("4. Side-by-Side", "#065F46", "Conflicts are shown side-by-side with exact outlet attribution — never averaged."),
    ]
    for col, (title, color, desc) in zip((c1, c2, c3, c4), steps_info):
        with col:
            st.markdown(
                f"""
                <div class="step-card">
                    <h4 style="margin:0 0 6px 0; color:{color};">{title}</h4>
                    <p style="font-size:0.84rem; color:#475569; margin:0;">{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("#### 🏷️ Relationship Labels")
    st.markdown(
        "- 🔴 **Conflict (`DISAGREE`)**: Outlets report incompatible figures or outcomes for the same event.\n"
        "- 🟢 **Agreement (`AGREE`)**: Outlets confirm the same fact or figure.\n"
        "- 🟡 **Timeline / Context Update (`DISCUSS`)**: Numbers differ due to evolving dates or scopes.\n"
        "- ⚪ **Unrelated / Insufficient (`UNRELATED` / `INSUFFICIENT_EVIDENCE`)**: Different topics or vague wording."
    )

    with st.expander("📰 View Raw Source Articles on Disk", expanded=False):
        raw_candidates = sorted(
            glob.glob(os.path.join(RAW_SOURCES_DIR, "*.txt"))
            + glob.glob(os.path.join(CUSTOM_SOURCES_DIR, "*.txt"))
        )
        raw_files = [
            f for f in raw_candidates
            if os.path.isfile(f) and not os.path.basename(f).startswith("README")
        ]
        if not raw_files:
            st.write("No source `.txt` files found.")
        else:
            file_tabs = st.tabs([os.path.basename(f) for f in raw_files])
            for tab, fp in zip(file_tabs, raw_files):
                with tab:
                    try:
                        with open(fp, "r", encoding="utf-8") as rf:
                            st.code(rf.read(), language="text")
                    except OSError as exc:
                        st.warning(f"Could not read {os.path.basename(fp)}: {exc}")


def main() -> None:
    """Render the complete Contradiction-Aware News Analysis dashboard."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="app-header">
            <h1 class="app-title">⚖️ Contradiction-Aware News Analysis</h1>
            <p class="app-subtitle">
                Compare news articles side-by-side — detect conflicting figures, confirmed agreements, and timeline updates with full source attribution.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    has_data, claims_list, clusters_dict, summary_md = _load_analysis_outputs()
    all_sources = sorted(
        set((c.get("source_name") or c.get("source_file") or "Unknown") for c in claims_list)
    )

    _render_sidebar(has_data)

    primary_tab, guide_tab = st.tabs(["🔍 Compare & Results", "📚 Claims Explorer & Viva Guide"])

    with primary_tab:
        _render_input_panel(has_data)
        if has_data:
            _render_results_section(claims_list, clusters_dict, summary_md, all_sources)
        else:
            st.info("👆 Select a sample or paste two articles above, then click **Analyze & Compare Articles** to see results.")

    with guide_tab:
        _render_explorer_and_guide(claims_list, all_sources)


if __name__ == "__main__":
    main()
