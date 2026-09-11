"""
Step 4: Interactive Streamlit Demo App
---------------------------------------
A rich, modern interactive dashboard to demo the Contradiction-Aware
Multi-Document Summarizer to faculty and reviewers.

Run: uv run streamlit run src/04_app.py
"""

import os
import json
import glob
import streamlit as st
import importlib.util

# Set Streamlit page configuration
st.set_page_config(
    page_title="Contradiction-Aware News Summarizer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern, polished look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        color: #F8FAFC;
        padding: 1.8rem 2.2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
        background: linear-gradient(90deg, #38BDF8, #818CF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .main-header p {
        color: #94A3B8;
        font-size: 1.05rem;
        margin: 0;
    }

    .source-tag {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 4px;
    }
    .source-tag-disputed {
        background-color: #FEE2E2;
        color: #991B1B;
        border: 1px solid #FCA5A5;
    }
    .source-tag-confirmed {
        background-color: #D1FAE5;
        color: #065F46;
        border: 1px solid #6EE7B7;
    }

    .conflict-callout {
        background-color: #FFF1F2;
        border: 1px dashed #F43F5E;
        border-radius: 8px;
        padding: 10px 14px;
        margin-top: 10px;
        color: #881337;
    }
</style>
""", unsafe_allow_html=True)


# Banner
st.markdown("""
<div class="main-header">
    <h1>⚖️ Contradiction-Aware News Summarizer</h1>
    <p>Multi-document cross-source verification engine: Segregating <b>Confirmed</b> facts, detecting <b>Disputed</b> contradictions side-by-side, and cataloging <b>Unconfirmed</b> claims.</p>
</div>
""", unsafe_allow_html=True)


# Sidebar controls
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/news.png", width=64)
    st.header("Pipeline Configuration")
    
    st.markdown("**LLM Provider:** Groq (`qwen/qwen3.8-27b`)")
    st.markdown("**Embedder:** `all-MiniLM-L6-v2`")
    
    distance_threshold = st.slider(
        "Clustering Distance Threshold",
        min_value=0.30,
        max_value=0.85,
        value=0.55,
        step=0.05,
        help="Higher values group broader topics together; lower values create more granular fact clusters."
    )

    st.markdown("---")
    st.subheader("Source Documents")
    
    upload_mode = st.radio(
        "Source Selection:",
        ["Use Default 2026 Assam Floods Sources", "Upload Custom .txt Files"],
        index=0
    )

    active_files = []
    if upload_mode == "Use Default 2026 Assam Floods Sources":
        default_files = sorted(glob.glob("data/raw_sources/source*.txt"))
        if default_files:
            st.success(f"Found {len(default_files)} built-in source files:")
            for f in default_files:
                st.caption(f"📄 `{os.path.basename(f)}`")
            active_files = default_files
        else:
            st.warning("Default sources not found in data/raw_sources/.")
    else:
        uploaded_files = st.file_uploader(
            "Upload 2 or more conflicting news articles (.txt)",
            type=["txt"],
            accept_multiple_files=True
        )
        if uploaded_files:
            os.makedirs("data/raw_sources", exist_ok=True)
            for uf in uploaded_files:
                save_path = os.path.join("data/raw_sources", uf.name)
                with open(save_path, "wb") as f:
                    f.write(uf.getbuffer())
                active_files.append(save_path)
            st.success(f"Loaded {len(active_files)} uploaded files.")

    st.markdown("---")
    run_button = st.button("🚀 Run Full Pipeline", type="primary", use_container_width=True)


# Helper function to run pipeline
def execute_pipeline(threshold: float):
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Step 1
    status_text.info("Step 1/3: Extracting atomic claims using Groq LLM...")
    progress_bar.progress(15)
    
    spec1 = importlib.util.spec_from_file_location("step1", "src/01_extract_claims.py")
    step1 = importlib.util.module_from_spec(spec1)
    spec1.loader.exec_module(step1)
    step1.main()
    
    progress_bar.progress(45)
    status_text.info("Step 2/3: Clustering facts with SentenceTransformer & classifying stances via Groq...")

    spec2 = importlib.util.spec_from_file_location("step2", "src/02_cluster_and_classify.py")
    step2 = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(step2)
    step2.run_pipeline_step2(distance_threshold=threshold)

    progress_bar.progress(85)
    status_text.info("Step 3/3: Synthesizing Contradiction-Aware Markdown Report...")

    spec3 = importlib.util.spec_from_file_location("step3", "src/03_generate_summary.py")
    step3 = importlib.util.module_from_spec(spec3)
    spec3.loader.exec_module(step3)
    step3.run_pipeline_step3()

    progress_bar.progress(100)
    status_text.success("✅ Analysis Complete! Results updated below.")
    st.toast("Pipeline finished successfully!", icon="🎉")


if run_button:
    if not active_files:
        st.error("Please provide at least 2 source files before running.")
    else:
        with st.spinner("Processing multi-document news corpus..."):
            try:
                execute_pipeline(distance_threshold)
            except Exception as e:
                st.error(f"Pipeline error: {e}")


# Main Display Section
tabs = st.tabs(["📊 Executive Summary", "🔍 Cluster & Stance Explorer", "📝 Raw Markdown Report", "📁 Extracted Claims Data"])

claims_file = "output/claims.json"
clusters_file = "output/clusters_with_stance.json"
summary_file = "output/final_summary.md"

has_results = os.path.exists(clusters_file) and os.path.exists(summary_file)

if has_results:
    with open(clusters_file, "r", encoding="utf-8") as f:
        clusters_data = json.load(f)

    # Categorize clusters cleanly
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

    # Tab 1: Executive Summary
    with tabs[0]:
        # Metric KPI cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Confirmed Fact Clusters", len(confirmed_list), help="Corroborated by 2+ independent sources with 0 disagreements")
        with col2:
            st.metric("Disputed Contradictions", len(disputed_list), help="Active contradictions detected between sources")
        with col3:
            st.metric("Unconfirmed / Developing", len(unconfirmed_list), help="Reported by single source or pending verification")

        st.markdown("---")

        # 1. Disputed Section (Highest Priority)
        st.subheader(f"🚨 Disputed Facts ({len(disputed_list)})")
        st.caption("Contradictory claims are juxtaposed side-by-side with full source traceability rather than being merged.")
        
        if not disputed_list:
            st.info("No contradictions detected across the provided documents.")
        else:
            for idx, c in enumerate(disputed_list, 1):
                sources = sorted(set(m["source_file"] for m in c["claims"]))
                
                with st.container(border=True):
                    # Header
                    tags_html = "".join([f'<span class="source-tag source-tag-disputed">{s}</span>' for s in sources])
                    st.markdown(f"### ⚡ Contradiction Cluster #{idx}", unsafe_allow_html=True)
                    st.markdown(f"**Involved Outlets:** {tags_html}", unsafe_allow_html=True)
                    
                    st.markdown("**Reported Perspectives (Side-by-Side):**")
                    for m in c["claims"]:
                        date_str = f" `({m['date']})`" if m.get("date") else ""
                        attr_str = f" — *Attributed to: {m['attribution']}*" if m.get("attribution") else ""
                        st.markdown(f"- **[{m['source_file']}]**{date_str}: \"{m['claim']}\"{attr_str}")

                    # Contradiction edges
                    disagree_edges = [e for e in c.get("edges", []) if e.get("stance") == "disagree"]
                    if disagree_edges:
                        st.markdown("**Detected Disagreements:**")
                        seen_pairs = set()
                        for e in disagree_edges:
                            pair_key = tuple(sorted([e["claim_a"], e["claim_b"]]))
                            if pair_key in seen_pairs:
                                continue
                            seen_pairs.add(pair_key)
                            
                            callout_html = (
                                f'<div class="conflict-callout">'
                                f'<b>⚡ DISAGREEMENT:</b><br>'
                                f'<b>[{e["source_a"]}]</b>: <i>"{e["claim_a"]}"</i><br>'
                                f'<b style="color: #64748B;">vs</b><br>'
                                f'<b>[{e["source_b"]}]</b>: <i>"{e["claim_b"]}"</i>'
                                f'</div>'
                            )
                            st.markdown(callout_html, unsafe_allow_html=True)

        st.markdown("---")

        # 2. Confirmed Section
        st.subheader(f"✅ Confirmed Facts ({len(confirmed_list)})")
        st.caption("Factual statements agreed upon and corroborated across multiple independent outlets.")
        
        if not confirmed_list:
            st.info("No multi-source corroborated facts found.")
        else:
            for idx, c in enumerate(confirmed_list, 1):
                sources = sorted(set(m["source_file"] for m in c["claims"]))
                
                with st.container(border=True):
                    tags_html = "".join([f'<span class="source-tag source-tag-confirmed">{s}</span>' for s in sources])
                    st.markdown(f"#### Fact #{idx}: {c['claims'][0]['claim']}")
                    st.markdown(f"**Corroborating Sources:** {tags_html}", unsafe_allow_html=True)
                    
                    st.markdown("**Source Statements:**")
                    for m in c["claims"]:
                        date_str = f" `({m['date']})`" if m.get("date") else ""
                        attr_str = f" — *Attr: {m['attribution']}*" if m.get("attribution") else ""
                        st.markdown(f"- **[{m['source_file']}]**{date_str}: \"{m['claim']}\"{attr_str}")

        st.markdown("---")

        # 3. Unconfirmed Section
        st.subheader(f"ℹ️ Unconfirmed & Developing Claims ({len(unconfirmed_list)})")
        with st.expander(f"Show {len(unconfirmed_list)} Single-Source or Developing Claims", expanded=False):
            for c in unconfirmed_list:
                for m in c["claims"]:
                    date_tag = f" `({m['date']})`" if m.get("date") else ""
                    attr_tag = f" [Attr: {m['attribution']}]" if m.get("attribution") else ""
                    st.markdown(f"- 📄 **`{m['source_file']}`**{date_tag}: {m['claim']}{attr_tag}")

    # Tab 2: Explorer
    with tabs[1]:
        st.subheader("Interactive Cluster & Stance Inspector")
        st.caption("Inspect how SentenceTransformer grouped claims and how Groq LLM classified cross-source pairs.")
        
        for cid, cdata in clusters_data.items():
            members = cdata["claims"]
            edges = cdata.get("edges", [])
            sources = set(m["source_file"] for m in members)
            
            has_disagree = any(e.get("stance") == "disagree" for e in edges)
            has_agree = any(e.get("stance") == "agree" for e in edges)
            
            status_label = "🔴 DISPUTED" if has_disagree else ("🟢 CONFIRMED" if (len(sources) > 1 and has_agree) else "⚪ UNCONFIRMED")
            
            with st.expander(f"Cluster ID {cid} | {status_label} | {len(members)} claims from {len(sources)} sources"):
                st.write("**Claims in this cluster:**")
                st.dataframe(members, use_container_width=True)
                
                if edges:
                    st.write("**Pairwise Cross-Source Stance Classifications:**")
                    st.dataframe(edges, use_container_width=True)
                else:
                    st.caption("No cross-source pairs to evaluate.")

    # Tab 3: Raw Markdown Report
    with tabs[2]:
        st.subheader("Generated Final Summary Markdown")
        with open(summary_file, "r", encoding="utf-8") as f:
            md_content = f.read()
        
        st.download_button(
            "📥 Download final_summary.md",
            data=md_content,
            file_name="final_summary.md",
            mime="text/markdown"
        )
        st.markdown(md_content)

    # Tab 4: Raw Claims Data
    with tabs[3]:
        st.subheader("Extracted Atomic Claims (output/claims.json)")
        if os.path.exists(claims_file):
            with open(claims_file, "r", encoding="utf-8") as f:
                claims_json = json.load(f)
            st.download_button(
                "📥 Download claims.json",
                data=json.dumps(claims_json, indent=2),
                file_name="claims.json",
                mime="application/json"
            )
            st.dataframe(claims_json, use_container_width=True)

else:
    with tabs[0]:
        st.info("👈 Click **'🚀 Run Full Pipeline'** in the sidebar to extract claims, cluster facts, classify contradictions, and generate the summary report!")
