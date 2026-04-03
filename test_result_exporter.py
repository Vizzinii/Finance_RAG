import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from eval_engine.result_exporter import (
    build_export_payload,
    flatten_results_for_csv,
    summarize_error_buckets
)


def test_result_exporter():
    print("Testing evaluation result exporter...")

    results = [
        {
            "id": 1,
            "mode": "overall",
            "question": "Q1",
            "candidate_answer": "A1",
            "expected_label": "positive",
            "predicted_label": "positive",
            "verdict": "hallucinated",
            "confidence": 0.9,
            "reason": "Wrong number.",
            "source_model": "weak-model",
            "source_type": "weak_model",
            "ground_truth": "GT1",
            "is_correct": True,
            "evidence": ["E1"],
            "unsupported_parts": ["U1"]
        },
        {
            "id": 2,
            "mode": "claim",
            "question": "Q2",
            "candidate_answer": "A2",
            "expected_label": "negative",
            "predicted_label": "uncertain",
            "verdict": "uncertain",
            "confidence": 0.0,
            "reason": "Missing evidence.",
            "source_model": "kb-grounded",
            "source_type": "grounded_answer",
            "ground_truth": "GT2",
            "is_correct": False,
            "claim_counts": {
                "supported": 0,
                "contradicted": 0,
                "insufficient_evidence": 1
            },
            "claim_results": [
                {
                    "claim": "C1",
                    "verdict": "insufficient_evidence",
                    "confidence": 0.6,
                    "reason": "Need more evidence.",
                    "evidence": []
                }
            ]
        }
    ]
    metrics = {"accuracy": 0.5, "precision": 1.0, "recall": 0.5, "f1": 0.67, "uncertain_rate": 0.5}

    flattened = flatten_results_for_csv(results)
    print("Flattened:", flattened)
    if len(flattened) == 2 and flattened[1]["claim_insufficient_evidence_count"] == 1:
        print("SUCCESS: CSV flattening works.")
    else:
        print("FAILURE: CSV flattening mismatch.")

    payload = build_export_payload(results, metrics, "claim", "demo_dataset", {"provider": "OpenAI"})
    payload_json = json.dumps(payload, ensure_ascii=False)
    print("Payload:", payload_json)
    if payload["dataset_name"] == "demo_dataset" and payload["metrics"]["accuracy"] == 0.5:
        print("SUCCESS: JSON payload building works.")
    else:
        print("FAILURE: JSON payload mismatch.")

    buckets = summarize_error_buckets(results)
    print("Buckets:", buckets)
    if len(buckets["incorrect"]) == 1 and len(buckets["uncertain"]) == 1:
        print("SUCCESS: Error bucket summary works.")
    else:
        print("FAILURE: Error bucket summary mismatch.")


if __name__ == "__main__":
    test_result_exporter()
