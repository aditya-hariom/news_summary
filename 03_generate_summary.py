"""
Step 3: Contradiction-Aware Summary Generator
------------------------------------------------
Reads output/clusters_with_stance.json and produces a structured summary:
- Confirmed Facts (clusters with 2+ independent sources, at least one agree edge, no disagree)
- Disputed Facts (clusters with at least one disagree edge) -- shown side by side,
  never merged into one sentence
- Unconfirmed / Developing (single-source clusters or multi-source discuss-only)

Run: python src/03_generate_summary.py
Output: output/final_summary.md
"""

import json
import os

def categorize_clusters(clusters: dict):
    confirmed, disputed, unconfirmed = [], [], []

    for cluster_id, data in clusters.items():
        members = data["claims"]
        edges = data.get("edges", [])

        sources = set(c["source_file"] for c in members)

        if len(sources) <= 1:
            unconfirmed.append(data)
            continue

        has_disagree = any(e.get("stance") == "disagree" for e in edges)
        has_agree = any(e.get("stance") == "agree" for e in edges)

        if has_disagree:
            disputed.append(data)
        elif has_agree:
            confirmed.append(data)
        else:
            # Multi-source cluster but only discuss/unrelated (not corroborated consensus)
            unconfirmed.append(data)

    return confirmed, disputed, unconfirmed


def render_markdown(confirmed, disputed, unconfirmed) -> str:
    lines = ["# Contradiction-Aware Multi-Document News Summary\n"]
    lines.append(f"**Generated Report** | Confirmed: {len(confirmed)} | Disputed: {len(disputed)} | Unconfirmed/Developing: {len(unconfirmed)}\n")
    lines.append("---\n")

    # 1. Disputed Facts (Highest Priority)
    lines.append("## Disputed Facts\n")
    lines.append("> *Sources actively disagree on these points (e.g. conflicting casualties, numbers, or statuses). Shown side-by-side with source attribution, never merged into one narrative.*\n")
    if not disputed:
        lines.append("_None detected in this batch._\n")
    for idx, c in enumerate(disputed, 1):
        sources = sorted(set(m["source_file"] for m in c["claims"]))
        lines.append(f"### Disputed Fact Cluster #{idx} (Sources: {', '.join(f'`{s}`' for s in sources)})\n")
        
        lines.append("**Conflicting Perspectives (Side-by-Side):**\n")
        for m in c["claims"]:
            date_str = f" (Reported: {m['date']})" if m.get("date") else ""
            attr_str = f" — *Attributed to: {m['attribution']}*" if m.get("attribution") else ""
            lines.append(f"- **[{m['source_file']}]**{date_str}: \"{m['claim']}\"{attr_str}")

        lines.append("\n**Contradiction Analysis:**")
        disagree_edges = [e for e in c["edges"] if e.get("stance") == "disagree"]
        # Deduplicate edges if any
        seen_edges = set()
        for e in disagree_edges:
            pair_key = tuple(sorted([e["claim_a"], e["claim_b"]]))
            if pair_key in seen_edges:
                continue
            seen_edges.add(pair_key)
            lines.append(
                f"- **DISAGREEMENT DETECTED**:\n"
                f"  - **[{e['source_a']}]**: \"{e['claim_a']}\"\n"
                f"  - *versus*\n"
                f"  - **[{e['source_b']}]**: \"{e['claim_b']}\""
            )
        lines.append("")

    # 2. Confirmed Facts
    lines.append("## Confirmed Facts\n")
    lines.append("> *Corroborated by 2 or more independent news sources with direct agreement and zero detected disagreements.*\n")
    if not confirmed:
        lines.append("_None found in this batch._\n")
    for idx, c in enumerate(confirmed, 1):
        sources = sorted(set(m["source_file"] for m in c["claims"]))
        representative_claim = c["claims"][0]["claim"]
        lines.append(f"### {idx}. {representative_claim}\n")
        lines.append(f"- **Corroborating Sources**: {', '.join(f'`{s}`' for s in sources)}")
        lines.append("- **Source Statements**:")
        for m in c["claims"]:
            date_str = f" ({m['date']})" if m.get("date") else ""
            attr_str = f" [Attr: {m['attribution']}]" if m.get("attribution") else ""
            lines.append(f"  - **{m['source_file']}**{date_str}: \"{m['claim']}\"{attr_str}")
        lines.append("")

    # 3. Unconfirmed Claims
    lines.append("## Unconfirmed & Developing Claims\n")
    lines.append("> *Reported by a single outlet or lacking multi-source agreement; pending cross-verification.*\n")
    if not unconfirmed:
        lines.append("_None found in this batch._\n")
    for c in unconfirmed:
        for m in c["claims"]:
            date_str = f" ({m['date']})" if m.get("date") else ""
            attr_str = f" [Attr: {m['attribution']}]" if m.get("attribution") else ""
            lines.append(f"- **[{m['source_file']}]**{date_str}: \"{m['claim']}\"{attr_str}")

    return "\n".join(lines)


def run_pipeline_step3():
    input_path = "output/clusters_with_stance.json"
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path} not found. Run Step 2 first.")

    with open(input_path, "r", encoding="utf-8") as f:
        clusters = json.load(f)

    confirmed, disputed, unconfirmed = categorize_clusters(clusters)
    markdown = render_markdown(confirmed, disputed, unconfirmed)

    os.makedirs("output", exist_ok=True)
    output_path = "output/final_summary.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    print(f"Done. Saved to {output_path}")
    print(f"\nPipeline Summary: Confirmed: {len(confirmed)} | Disputed: {len(disputed)} | Unconfirmed: {len(unconfirmed)}")
    return confirmed, disputed, unconfirmed, markdown


def main():
    run_pipeline_step3()


if __name__ == "__main__":
    main()
