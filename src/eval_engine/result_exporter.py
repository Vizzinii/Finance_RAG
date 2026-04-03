import json
from typing import Any, Dict, List


def _serialize_claim_results(claim_results: List[Dict[str, Any]]) -> str:
    return json.dumps(claim_results, ensure_ascii=False)


def flatten_results_for_csv(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flattened = []
    for result in results:
        row = {
            "id": result.get("id"),
            "mode": result.get("mode", ""),
            "question": result.get("question", ""),
            "candidate_answer": result.get("candidate_answer", ""),
            "expected_label": result.get("expected_label", ""),
            "predicted_label": result.get("predicted_label", ""),
            "verdict": result.get("verdict", ""),
            "confidence": result.get("confidence", 0.0),
            "reason": result.get("reason", ""),
            "source_model": result.get("source_model", ""),
            "source_type": result.get("source_type", ""),
            "ground_truth": result.get("ground_truth", ""),
            "is_correct": result.get("is_correct", False),
            "evidence": " | ".join(result.get("evidence", [])),
            "unsupported_parts": " | ".join(result.get("unsupported_parts", []))
        }

        claim_counts = result.get("claim_counts", {})
        row["claim_supported_count"] = claim_counts.get("supported", 0)
        row["claim_contradicted_count"] = claim_counts.get("contradicted", 0)
        row["claim_insufficient_evidence_count"] = claim_counts.get("insufficient_evidence", 0)
        row["claim_results_json"] = _serialize_claim_results(result.get("claim_results", []))
        flattened.append(row)

    return flattened


def build_export_payload(
    results: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    mode: str,
    dataset_name: str = "",
    config_snapshot: Dict[str, Any] = None
) -> Dict[str, Any]:
    return {
        "dataset_name": dataset_name,
        "mode": mode,
        "metrics": metrics,
        "config": config_snapshot or {},
        "results": results
    }


def summarize_error_buckets(results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    return {
        "incorrect": [result for result in results if not result.get("is_correct", False)],
        "uncertain": [result for result in results if result.get("predicted_label") == "uncertain"],
        "false_positive": [
            result
            for result in results
            if result.get("expected_label") == "negative" and result.get("predicted_label") == "positive"
        ],
        "false_negative": [
            result
            for result in results
            if result.get("expected_label") == "positive" and result.get("predicted_label") != "positive"
        ]
    }
