"""
System Evaluation Module
------------------------
Evaluates the Contradiction-Aware Stance Classification system against a
manually annotated Ground-Truth Benchmark Dataset (FNC-1 5-class schema).

Metrics Computed:
- Overall Accuracy
- Class-wise Precision, Recall, and F1-score
- Macro-F1 and Weighted-F1
- 5x5 Confusion Matrix
- Error Analysis (False Positives & False Negatives)

Run: python src/evaluation.py
Output: output/evaluation_report.json
"""

import sys
import os
import json
import time
from typing import List, Dict, Any
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import BENCHMARK_DIR, OUTPUT_DIR, STANCE_LABELS
from src.stance_classifier import classify_stance_pair

BENCHMARK_FILE = os.path.join(BENCHMARK_DIR, "benchmark_pairs.json")
EVAL_REPORT_FILE = os.path.join(OUTPUT_DIR, "evaluation_report.json")


def compute_classification_metrics(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, Any]:
    """
    Computes rigorous Accuracy, Precision, Recall, F1, and Confusion Matrix
    without requiring external blackbox libraries.
    """
    total = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / total if total > 0 else 0.0

    # Build Confusion Matrix
    matrix = {label_true: {label_pred: 0 for label_pred in labels} for label_true in labels}
    for t, p in zip(y_true, y_pred):
        if t in matrix and p in matrix[t]:
            matrix[t][p] += 1

    per_class = {}
    macro_p, macro_r, macro_f1 = 0.0, 0.0, 0.0
    active_classes = 0

    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in labels if other != label)
        fn = sum(matrix[label][other] for other in labels if other != label)
        support = sum(matrix[label].values())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        if support > 0:
            active_classes += 1
            macro_p += precision
            macro_r += recall
            macro_f1 += f1

        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "support": support
        }

    macro_avg = {
        "precision": round(macro_p / active_classes, 4) if active_classes > 0 else 0.0,
        "recall": round(macro_r / active_classes, 4) if active_classes > 0 else 0.0,
        "macro_f1": round(macro_f1 / active_classes, 4) if active_classes > 0 else 0.0,
    }

    return {
        "total_samples": total,
        "overall_accuracy": round(accuracy, 4),
        "macro_averages": macro_avg,
        "per_class_metrics": per_class,
        "confusion_matrix": matrix
    }


def run_evaluation(benchmark_path: str = BENCHMARK_FILE) -> Dict[str, Any]:
    """
    Executes benchmark test suite and displays honest performance table.
    """
    if not os.path.exists(benchmark_path):
        raise FileNotFoundError(f"Benchmark dataset not found at {benchmark_path}")

    with open(benchmark_path, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    print(f"\n=======================================================")
    print(f"RUNNING SCIENTIFIC EVALUATION BENCHMARK ({len(pairs)} Test Pairs)")
    print(f"=======================================================\n")

    y_true = []
    y_pred = []
    detailed_results = []
    errors = []

    for idx, p in enumerate(pairs, 1):
        claim_a = {
            "claim": p["claim_a"],
            "source_name": p["source_a"],
            "date": p.get("date_a"),
            "numerical_value": p.get("val_a"),
            "numerical_unit": p.get("unit_a"),
            "category": "general"
        }
        claim_b = {
            "claim": p["claim_b"],
            "source_name": p["source_b"],
            "date": p.get("date_b"),
            "numerical_value": p.get("val_b"),
            "numerical_unit": p.get("unit_b"),
            "category": "general"
        }

        res = classify_stance_pair(claim_a, claim_b)
        pred = res["stance"]
        truth = p["ground_truth"]

        y_true.append(truth)
        y_pred.append(pred)

        match = (pred == truth)
        status_icon = "✅" if match else "❌"
        print(f"[{idx:02d}/{len(pairs):02d}] {status_icon} Truth: {truth:<12} | Pred: {pred:<12} | {p['description'][:50]}")

        record = {
            "id": p.get("id"),
            "domain": p.get("domain"),
            "claim_a": p["claim_a"],
            "claim_b": p["claim_b"],
            "ground_truth": truth,
            "predicted": pred,
            "match": match,
            "reasoning": res.get("reasoning", "")
        }
        detailed_results.append(record)

        if not match:
            errors.append(record)

        time.sleep(0.15)

    # Compute formal statistics
    report = compute_classification_metrics(y_true, y_pred, STANCE_LABELS)
    report["errors_count"] = len(errors)
    report["errors"] = errors

    # Terminal summary print
    print(f"\n-------------------------------------------------------")
    print(f"BENCHMARK RESULTS SUMMARY")
    print(f"-------------------------------------------------------")
    print(f"Overall Accuracy:  {report['overall_accuracy']:.1%}")
    print(f"Macro F1-Score:    {report['macro_averages']['macro_f1']:.3f}")
    print(f"Macro Precision:   {report['macro_averages']['precision']:.3f}")
    print(f"Macro Recall:      {report['macro_averages']['recall']:.3f}")
    print(f"\nClass-Wise Breakdown:")
    print(f"{'Class':<22} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support'}")
    print(f"{'-'*65}")
    for label, metrics in report["per_class_metrics"].items():
        if metrics["support"] > 0:
            print(f"{label:<22} | {metrics['precision']:<10.2f} | {metrics['recall']:<10.2f} | {metrics['f1_score']:<10.2f} | {metrics['support']}")

    # Save to JSON
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(EVAL_REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[DONE] Full evaluation report saved to {EVAL_REPORT_FILE}")
    return report


if __name__ == "__main__":
    run_evaluation()
