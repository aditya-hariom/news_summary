"""
Contradiction-Aware News Summarization Engine
----------------------------------------------
Generates a structured, evidence-backed multi-document synthesis report:
1. Event Overview (Neutral synopsis)
2. Corroborated Claims (Multi-source agreement with responsible attribution)
3. Disputed Facts (Active cross-source contradictions shown side-by-side)
4. Chronological & Developing Updates (Temporal progression)
5. Unverified / Single-Source Reports (Pending independent confirmation)
6. Unknowns & Insufficient Evidence (Information gaps)

Strict Epistemic Modesty: Never merges conflicting figures into one narrative;
never asserts majority consensus as absolute objective truth.
"""

import sys
import os
import json
from typing import Dict, List, Any, Tuple, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import CLUSTERS_FILE, SUMMARY_FILE, CLAIMS_FILE


def categorize_clusters(clusters: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Categorizes clustered claims based on stance edges and source independence:
    Returns: (corroborated, disputed, developing, unverified)
    """
    corroborated = []
    disputed = []
    developing = []
    unverified = []

    for cid, cdata in clusters.items():
        claims = cdata["claims"]
        edges = cdata.get("edges", [])
        sources = set(c.get("source_name") or c.get("source_file") for c in claims)

        if len(sources) <= 1:
            unverified.append(cdata)
            continue

        has_disagree = any(e.get("stance") == "DISAGREE" for e in edges)
        has_agree = any(e.get("stance") == "AGREE" for e in edges)
        has_discuss = any(e.get("stance") == "DISCUSS" for e in edges)

        if has_disagree:
            disputed.append(cdata)
        elif has_agree:
            corroborated.append(cdata)
        elif has_discuss:
            developing.append(cdata)
        else:
            unverified.append(cdata)

    return corroborated, disputed, developing, unverified


def generate_event_overview(
    claims: List[Dict[str, Any]],
    corroborated: Optional[List[Dict[str, Any]]] = None,
    disputed: Optional[List[Dict[str, Any]]] = None,
    developing: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Return a short, plain-English overview of the checked articles."""
    if not claims:
        return "No news facts were provided to summarize."

    sources = sorted(set(c.get("source_name") or c.get("source_file") or "Unknown" for c in claims))
    source_list = ", ".join(f"`{s}`" for s in sources)
    n_conflicts = len(disputed) if disputed is not None else 0
    n_agreed = len(corroborated) if corroborated is not None else 0
    n_updates = len(developing) if developing is not None else 0

    lines = [
        f"Checked **{len(claims)} facts** across **{len(sources)} news sources** ({source_list}).",
        f"- **{n_conflicts} conflict(s)** where reports give different numbers or details.",
        f"- **{n_agreed} agreed fact(s)** confirmed by 2 or more sources.",
    ]
    if n_updates:
        lines.append(f"- **{n_updates} timeline update(s)** where figures changed across different dates.")
    return "\n".join(lines)


def _format_claim_bullet(claim: Dict[str, Any]) -> str:
    """Format a claim bullet with source, optional date, and distinct attribution."""
    src = str(claim.get("source_name") or claim.get("source_file") or "Unknown")
    d_str = f" *({claim['date']})*" if claim.get("date") else ""
    attr_val = str(claim.get("attribution") or "").strip()
    via = f" *(via {attr_val})*" if attr_val and attr_val.lower() != src.lower() else ""
    return f"- **{src}**{d_str}{via}: \"{claim['claim']}\""


def render_markdown_report(
    corroborated: List[Dict[str, Any]],
    disputed: List[Dict[str, Any]],
    developing: List[Dict[str, Any]],
    unverified: List[Dict[str, Any]],
    all_claims: List[Dict[str, Any]]
) -> str:
    """Render a short, plain-English markdown report comparing all news sources."""
    sources = sorted(set(c.get("source_name") or c.get("source_file") or "Unknown" for c in all_claims))
    lines = [
        "# News Comparison Report\n",
        (
            f"**Sources Checked:** {len(sources)} | "
            f"**Conflicts:** {len(disputed)} | "
            f"**Agreed Facts:** {len(corroborated)} | "
            f"**Updates Over Time:** {len(developing)} | "
            f"**Only 1 Source:** {len(unverified)}\n"
        ),
        "---\n",
        "## Quick Overview\n",
        generate_event_overview(all_claims, corroborated, disputed, developing) + "\n",
    ]

    # 1. Conflicting Facts (Side-by-Side)
    lines.append("## 1. Conflicting Facts (Side-by-Side)\n")
    lines.append("> *These reports disagree. Both sides are shown clearly without guessing who is right.*\n")

    if not disputed:
        lines.append("_No conflicting facts were found._\n")
    else:
        conflict_num = 0
        for cluster in disputed:
            topic_name = cluster.get("topic", "Fact").replace("_", " ").title()
            disagree_edges = [e for e in cluster.get("edges", []) if e.get("stance") == "DISAGREE"]
            seen_pairs = set()
            for e in disagree_edges:
                pair_key = tuple(sorted([e.get("claim_a", ""), e.get("claim_b", "")]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                conflict_num += 1
                attr = e.get("conflicting_attribute") or topic_name
                date_a = f" *({e['date_a']})*" if e.get("date_a") else ""
                date_b = f" *({e['date_b']})*" if e.get("date_b") else ""
                lines.append(f"### Conflict {conflict_num}: {attr}\n")
                lines.append(f"- **{e.get('source_a')}**{date_a}: \"{e.get('claim_a')}\"")
                lines.append(f"- **{e.get('source_b')}**{date_b}: \"{e.get('claim_b')}\"")
                if e.get("reasoning"):
                    lines.append(f"- **Why they differ:** {e['reasoning']}")
                lines.append("")

    # 2. Agreed Facts
    lines.append("## 2. Agreed Facts\n")
    lines.append("> *Facts reported the same way by 2 or more news sources.*\n")
    if not corroborated:
        lines.append("_No facts were confirmed by multiple sources._\n")
    else:
        for cluster in corroborated:
            claims = cluster.get("claims", [])
            if not claims:
                continue
            c_sources = sorted(set(c.get("source_name") or c.get("source_file") or "Unknown" for c in claims))
            lines.append(f"- **\"{claims[0]['claim']}\"** — *Sources: {', '.join(f'`{s}`' for s in c_sources)}*")
        lines.append("")

    # 3. Updates Over Time
    lines.append("## 3. Updates Over Time\n")
    lines.append("> *Numbers or details that changed as reports came out on different dates.*\n")
    if not developing:
        lines.append("_No date-based updates found._\n")
    else:
        for cluster in developing:
            topic = cluster.get("topic", "Update").replace("_", " ").title()
            lines.append(f"### {topic}\n")
            for c in cluster.get("claims", []):
                lines.append(_format_claim_bullet(c))
            lines.append("")

    # 4. Reported by Only One Source
    lines.append("## 4. Reported by Only One Source\n")
    lines.append("> *Mentioned by only one source so far — not checked by other reports yet.*\n")
    if not unverified:
        lines.append("_Every fact was covered by multiple sources._\n")
    else:
        for cluster in unverified:
            for c in cluster.get("claims", []):
                lines.append(_format_claim_bullet(c))
        lines.append("")

    return "\n".join(lines)


def run_summary_pipeline(
    clusters_path: str = CLUSTERS_FILE,
    claims_path: str = CLAIMS_FILE,
    output_path: str = SUMMARY_FILE
) -> Tuple[List[Any], List[Any], List[Any], str]:
    """Generates and writes the final contradiction-aware summary report."""
    if not os.path.exists(clusters_path):
        raise FileNotFoundError(f"{clusters_path} not found. Run Step 2 (Clustering & Stance) first.")

    with open(clusters_path, "r", encoding="utf-8") as f:
        clusters = json.load(f)

    all_claims = []
    if os.path.exists(claims_path):
        with open(claims_path, "r", encoding="utf-8") as f:
            all_claims = json.load(f)

    print("Categorizing fact clusters by stance agreement & conflicts...")
    corroborated, disputed, developing, unverified = categorize_clusters(clusters)

    print("Generating comprehensive Contradiction-Aware Markdown Report...")
    markdown_report = render_markdown_report(corroborated, disputed, developing, unverified, all_claims)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown_report)

    print(f"\n[DONE] Summary Report successfully saved to {output_path}")
    print(f"Summary Breakdown: Corroborated: {len(corroborated)} | Disputed: {len(disputed)} | Developing: {len(developing)} | Single-Source: {len(unverified)}")

    return corroborated, disputed, unverified, markdown_report


if __name__ == "__main__":
    run_summary_pipeline()
