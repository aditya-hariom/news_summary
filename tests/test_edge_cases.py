"""
Automated Edge-Case & Runtime Verification Test Suite
-----------------------------------------------------
Tests all 10 edge cases defined in the Final Verification Phase:
1. Identical numbers and meanings
2. Clearly contradictory numbers
3. Different units (e.g. deaths vs injured)
4. Different time periods (temporal progression)
5. Later report correcting earlier number (revision)
6. Two unrelated claims
7. Insufficient information to determine stance
8. Single-article input
9. Empty input
10. Invalid / malformed input
"""

import sys
import os
import json
import traceback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.stance_classifier import classify_stance_pair, deterministic_stance_check
from src.clustering import cluster_claims
from src.extraction import extract_claims_from_text, parse_claims_json
from src.summarizer import categorize_clusters, render_markdown_report


def run_edge_case_tests():
    print("\n=======================================================")
    print("STARTING EDGE CASE AND RUNTIME VERIFICATION SUITE")
    print("=======================================================\n")
    results = {}

    # -------------------------------------------------------------------------
    # CASE 1: Two claims with identical numbers and meanings
    # -------------------------------------------------------------------------
    c1_a = {
        "claim": "52 deaths have been recorded across the district.",
        "source_name": "Outlet A",
        "date": "Thursday",
        "numerical_value": 52,
        "numerical_unit": "deaths",
        "category": "casualties"
    }
    c1_b = {
        "claim": "The confirmed death toll stands at 52.",
        "source_name": "Outlet B",
        "date": "Thursday",
        "numerical_value": 52,
        "numerical_unit": "deaths",
        "category": "casualties"
    }
    res1 = classify_stance_pair(c1_a, c1_b)
    pass1 = (res1["stance"] == "AGREE")
    print(f"Case 1 (Identical numbers & meanings): {'PASS' if pass1 else 'FAIL'} -> Stance: {res1['stance']}")
    results["Case 1"] = {"passed": pass1, "stance": res1["stance"], "reasoning": res1.get("reasoning")}

    # -------------------------------------------------------------------------
    # CASE 2: Two claims with clearly contradictory numbers (same timeframe)
    # -------------------------------------------------------------------------
    c2_a = {
        "claim": "The flood death toll climbed to 82.",
        "source_name": "Times of India",
        "date": "2 August 2026",
        "numerical_value": 82,
        "numerical_unit": "deaths",
        "category": "casualties"
    }
    c2_b = {
        "claim": "The official death toll stands at 47.",
        "source_name": "MapsOfIndia",
        "date": None,
        "numerical_value": 47,
        "numerical_unit": "deaths",
        "category": "casualties"
    }
    c2_b["date"] = None
    res2 = classify_stance_pair(c2_a, c2_b)
    pass2 = (res2["stance"] == "DISAGREE")
    print(f"Case 2 (Clearly contradictory numbers): {'PASS' if pass2 else 'FAIL'} -> Stance: {res2['stance']}")
    results["Case 2"] = {"passed": pass2, "stance": res2["stance"], "attribute": res2.get("conflicting_attribute"), "reasoning": res2.get("reasoning")}

    # -------------------------------------------------------------------------
    # CASE 3: Claims with different units (e.g. deaths vs injured)
    # -------------------------------------------------------------------------
    c3_a = {
        "claim": "Five people died in the accident.",
        "source_name": "Outlet A",
        "numerical_value": 5,
        "numerical_unit": "deaths",
        "category": "casualties"
    }
    c3_b = {
        "claim": "Twenty people were injured in the accident.",
        "source_name": "Outlet B",
        "numerical_value": 20,
        "numerical_unit": "injured",
        "category": "casualties"
    }
    res3 = classify_stance_pair(c3_a, c3_b)
    pass3 = (res3["stance"] != "DISAGREE")  # Must NOT be flagged as contradiction!
    print(f"Case 3 (Different units: deaths vs injured): {'PASS' if pass3 else 'FAIL'} -> Stance: {res3['stance']} (Expected: NOT DISAGREE)")
    results["Case 3"] = {"passed": pass3, "stance": res3["stance"], "reasoning": res3.get("reasoning")}

    # -------------------------------------------------------------------------
    # CASE 4: Claims referring to different time periods (chronological progression)
    # -------------------------------------------------------------------------
    c4_a = {
        "claim": "The flood toll was reported at 66 on 26 July 2026.",
        "source_name": "The Hindu",
        "date": "26 July 2026",
        "numerical_value": 66,
        "numerical_unit": "deaths",
        "category": "casualties"
    }
    c4_b = {
        "claim": "The flood death toll reached 100 as of 10 August 2026.",
        "source_name": "NPR",
        "date": "10 August 2026",
        "numerical_value": 100,
        "numerical_unit": "deaths",
        "category": "casualties"
    }
    res4 = classify_stance_pair(c4_a, c4_b)
    pass4 = (res4["stance"] == "DISCUSS")
    print(f"Case 4 (Different time periods / progression): {'PASS' if pass4 else 'FAIL'} -> Stance: {res4['stance']}")
    results["Case 4"] = {"passed": pass4, "stance": res4["stance"], "reasoning": res4.get("reasoning")}

    # -------------------------------------------------------------------------
    # CASE 5: A later report correcting an earlier number (downward revision)
    # -------------------------------------------------------------------------
    c5_a = {
        "claim": "Initial wire reports stated 288 passengers died in the train derailment on 3 June.",
        "source_name": "Wire Agency",
        "date": "3 June",
        "numerical_value": 288,
        "numerical_unit": "deaths",
        "category": "casualties"
    }
    c5_b = {
        "claim": "Officials revised the death count downwards to 275 after double counting was eliminated on 4 June.",
        "source_name": "Ministry Bulletin",
        "date": "4 June",
        "numerical_value": 275,
        "numerical_unit": "deaths",
        "category": "casualties"
    }
    res5 = classify_stance_pair(c5_a, c5_b)
    # Both DISAGREE (due to revision conflict) or DISCUSS (temporal change) are informative; let's record exact reasoning
    print(f"Case 5 (Correction / Revision of number): Stance -> {res5['stance']} | Reasoning: {res5.get('reasoning')[:70]}...")
    results["Case 5"] = {"stance": res5["stance"], "reasoning": res5.get("reasoning")}

    # -------------------------------------------------------------------------
    # CASE 6: Two unrelated claims
    # -------------------------------------------------------------------------
    c6_a = {
        "claim": "52 deaths were confirmed across the district.",
        "source_name": "Outlet A",
        "category": "casualties"
    }
    c6_b = {
        "claim": "Relief teams distributed 5,000 food packets in relief camps.",
        "source_name": "Red Cross",
        "category": "rescue_relief"
    }
    res6 = classify_stance_pair(c6_a, c6_b)
    pass6 = (res6["stance"] == "UNRELATED")
    print(f"Case 6 (Unrelated claims across categories): {'PASS' if pass6 else 'FAIL'} -> Stance: {res6['stance']}")
    results["Case 6"] = {"passed": pass6, "stance": res6["stance"], "reasoning": res6.get("reasoning")}

    # -------------------------------------------------------------------------
    # CASE 7: Claims with insufficient information to determine stance
    # -------------------------------------------------------------------------
    c7_a = {
        "claim": "Heavy damages were reported in several remote villages.",
        "source_name": "Local Correspondent",
        "date": None,
        "numerical_value": None,
        "category": "damage"
    }
    c7_b = {
        "claim": "Three houses developed minor cracks following the earthquake tremor.",
        "source_name": "District Magistrate",
        "date": "Wednesday",
        "numerical_value": 3,
        "category": "damage"
    }
    res7 = classify_stance_pair(c7_a, c7_b)
    print(f"Case 7 (Insufficient info / ambiguous qualifiers): Stance -> {res7['stance']} | Reasoning: {res7.get('reasoning')[:70]}...")
    results["Case 7"] = {"stance": res7["stance"], "reasoning": res7.get("reasoning")}

    # -------------------------------------------------------------------------
    # CASE 8: Single-article input
    # -------------------------------------------------------------------------
    single_claim = [{
        "claim_id": "clm_single_01",
        "article_id": "single_art",
        "source_name": "Sole Outlet",
        "claim": "A new solar power plant was inaugurated today.",
        "category": "energy"
    }]
    clusters_single, _ = cluster_claims(single_claim)
    # Check if summarizer handles single cluster without crashing
    clusters_single_dict = {
        "0": {
            "cluster_id": 0,
            "topic": "energy",
            "claims": single_claim,
            "edges": []
        }
    }
    corrob, disp, dev, unconf = categorize_clusters(clusters_single_dict)
    pass8 = (len(disp) == 0 and len(corrob) == 0 and len(unconf) == 1)
    md_single = render_markdown_report(corrob, disp, dev, unconf, single_claim)
    print(f"Case 8 (Single-article input): {'PASS' if pass8 and len(md_single) > 50 else 'FAIL'} -> Unconfirmed: {len(unconf)}, Disputed: {len(disp)}")
    results["Case 8"] = {"passed": pass8, "unconfirmed_count": len(unconf), "disputed_count": len(disp)}

    # -------------------------------------------------------------------------
    # CASE 9: Empty input
    # -------------------------------------------------------------------------
    empty_claims = []
    clusters_empty, _ = cluster_claims(empty_claims)
    corrob_e, disp_e, dev_e, unconf_e = categorize_clusters(clusters_empty)
    md_empty = render_markdown_report(corrob_e, disp_e, dev_e, unconf_e, empty_claims)
    pass9 = (clusters_empty == {} and len(md_empty) > 20)
    print(f"Case 9 (Empty input handled safely): {'PASS' if pass9 else 'FAIL'} -> Clusters: {len(clusters_empty)}")
    results["Case 9"] = {"passed": pass9}

    # -------------------------------------------------------------------------
    # CASE 10: Invalid / malformed JSON extraction test
    # -------------------------------------------------------------------------
    malformed_raw_llm_outputs = [
        "```json\n{\"claims\": [{\"claim\": \"Valid claim\"}]}\n```",
        "Random text before {\"claims\": [{\"claim\": \"Regex extracted claim\"}]} random text after",
        "[{\"claim\": \"List formatted claim\"}]",
    ]
    all_parsed = True
    for text in malformed_raw_llm_outputs:
        try:
            parsed = parse_claims_json(text)
            if not parsed or not isinstance(parsed, list):
                all_parsed = False
        except Exception:
            all_parsed = False

    # Also test completely unparseable input to ensure clean ValueError
    threw_expected_error = False
    try:
        parse_claims_json("This is pure garbage without any JSON structure at all.")
    except ValueError:
        threw_expected_error = True

    pass10 = all_parsed and threw_expected_error
    print(f"Case 10 (Malformed JSON resilience): {'PASS' if pass10 else 'FAIL'}")
    results["Case 10"] = {"passed": pass10}

    print("\n-------------------------------------------------------")
    print("EDGE CASE TEST SUITE COMPLETED")
    print("-------------------------------------------------------\n")
    return results


if __name__ == "__main__":
    run_edge_case_tests()
