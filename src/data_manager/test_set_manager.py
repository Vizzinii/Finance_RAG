import json
import os
from copy import deepcopy
from typing import Any, Dict, List


class TestSetManager:
    DEFAULT_DATASET = {
        "dataset_name": "default_eval_set",
        "kb_version": "",
        "retrieval_config": {},
        "samples": []
    }

    def __init__(self, data_path: str = "./data/test_set.json"):
        self.data_path = data_path
        self.current_dataset = self.load_data()

    def _empty_dataset(self) -> Dict[str, Any]:
        return deepcopy(self.DEFAULT_DATASET)

    def _normalize_sample(self, sample: Dict[str, Any], default_id: int) -> Dict[str, Any]:
        return {
            "id": sample.get("id", default_id),
            "question": sample.get("question", "").strip(),
            "candidate_answer": sample.get("candidate_answer", "").strip(),
            "label": str(sample.get("label", "negative")).strip().lower() or "negative",
            "source_model": sample.get("source_model", "").strip(),
            "source_type": sample.get("source_type", "manual").strip(),
            "reference_docs": sample.get("reference_docs", []) or [],
            "ground_truth": sample.get("ground_truth", "").strip(),
            "notes": sample.get("notes", "").strip()
        }

    def _normalize_dataset(self, data: Any) -> Dict[str, Any]:
        dataset = self._empty_dataset()

        if isinstance(data, list):
            raw_samples = data
        elif isinstance(data, dict):
            dataset["dataset_name"] = data.get("dataset_name", dataset["dataset_name"])
            dataset["kb_version"] = data.get("kb_version", "")
            dataset["retrieval_config"] = data.get("retrieval_config", {}) or {}
            raw_samples = data.get("samples", [])
        else:
            raw_samples = []

        dataset["samples"] = [
            self._normalize_sample(sample, default_id=index)
            for index, sample in enumerate(raw_samples, start=1)
            if isinstance(sample, dict)
        ]
        return dataset

    def validate_dataset(self, dataset: Dict[str, Any]) -> List[str]:
        errors = []
        for sample in dataset.get("samples", []):
            if not sample.get("question"):
                errors.append(f"Sample {sample.get('id', '?')} is missing question.")
            if not sample.get("candidate_answer"):
                errors.append(f"Sample {sample.get('id', '?')} is missing candidate_answer.")
            if sample.get("label") not in {"positive", "negative"}:
                errors.append(
                    f"Sample {sample.get('id', '?')} has invalid label '{sample.get('label')}'."
                )
        return errors

    def load_data(self) -> Dict[str, Any]:
        """Load test set from JSON file, keeping backward compatibility with legacy lists."""
        if not os.path.exists(self.data_path):
            return self._empty_dataset()

        try:
            with open(self.data_path, "r", encoding="utf-8") as file:
                data = json.load(file)
        except json.JSONDecodeError:
            return self._empty_dataset()

        return self._normalize_dataset(data)

    def save_data(self):
        """Save current dataset to file."""
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        with open(self.data_path, "w", encoding="utf-8") as file:
            json.dump(self.current_dataset, file, ensure_ascii=False, indent=2)

    def import_dataset(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        normalized_dataset = self._normalize_dataset(dataset)
        errors = self.validate_dataset(normalized_dataset)
        if errors:
            raise ValueError("Invalid dataset: " + " ".join(errors))

        self.current_dataset = normalized_dataset
        self.save_data()
        return self.current_dataset

    def import_json_text(self, json_text: str) -> Dict[str, Any]:
        dataset = json.loads(json_text)
        return self.import_dataset(dataset)

    def import_json_file(self, file_path: str) -> Dict[str, Any]:
        with open(file_path, "r", encoding="utf-8") as file:
            return self.import_json_text(file.read())

    def get_dataset(self) -> Dict[str, Any]:
        return self.current_dataset

    def add_case(
        self,
        question: str,
        candidate_answer: str,
        label: str = "negative",
        source_model: str = "",
        source_type: str = "manual",
        ground_truth: str = "",
        reference_docs: List[str] = None,
        notes: str = ""
    ) -> Dict[str, Any]:
        """Add a new evaluation sample."""
        sample = self._normalize_sample(
            {
                "id": len(self.current_dataset["samples"]) + 1,
                "question": question,
                "candidate_answer": candidate_answer,
                "label": label,
                "source_model": source_model,
                "source_type": source_type,
                "ground_truth": ground_truth,
                "reference_docs": reference_docs or [],
                "notes": notes
            },
            default_id=len(self.current_dataset["samples"]) + 1
        )

        errors = self.validate_dataset({"samples": [sample]})
        if errors:
            raise ValueError("Invalid sample: " + " ".join(errors))

        self.current_dataset["samples"].append(sample)
        self.save_data()
        return sample

    def get_all_cases(self) -> List[Dict[str, Any]]:
        return self.current_dataset["samples"]

    def delete_case(self, case_id: int):
        """Delete a case by ID."""
        self.current_dataset["samples"] = [
            case for case in self.current_dataset["samples"] if case["id"] != case_id
        ]
        self.save_data()
