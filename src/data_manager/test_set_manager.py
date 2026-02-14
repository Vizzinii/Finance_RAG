import json
import os
from typing import List, Dict, Any

class TestSetManager:
    def __init__(self, data_path: str = "./data/test_set.json"):
        self.data_path = data_path
        self.current_dataset = self.load_data()

    def load_data(self) -> List[Dict[str, Any]]:
        """Load test set from JSON file."""
        if not os.path.exists(self.data_path):
            return []
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []

    def save_data(self):
        """Save current dataset to file."""
        with open(self.data_path, 'w', encoding='utf-8') as f:
            json.dump(self.current_dataset, f, ensure_ascii=False, indent=2)

    def add_case(self, question: str, ground_truth: str = "", reference_docs: list = None):
        """Add a new test case."""
        case = {
            "id": len(self.current_dataset) + 1,
            "question": question,
            "ground_truth": ground_truth,
            "reference_docs": reference_docs if reference_docs else []
        }
        self.current_dataset.append(case)
        self.save_data()
        return case

    def get_all_cases(self) -> List[Dict[str, Any]]:
        return self.current_dataset

    def delete_case(self, case_id: int):
        """Delete a case by ID."""
        self.current_dataset = [c for c in self.current_dataset if c['id'] != case_id]
        self.save_data()
