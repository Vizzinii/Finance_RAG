import json
import os
import shutil
import sys
import tempfile

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data_manager.test_set_manager import TestSetManager
from eval_engine.prompt_manager import PromptTemplateManager


def test_prompt_manager():
    print("Testing prompt template manager...")

    temp_dir = tempfile.mkdtemp(prefix="prompt_templates_", dir="data")
    try:
        manager = PromptTemplateManager(template_dir=temp_dir)
        default_prompts = manager.get_default_prompts()

        print("Default templates:", manager.list_templates())
        if manager.list_templates() == ["default"]:
            print("SUCCESS: Default template is available.")
        else:
            print("FAILURE: Default template listing mismatch.")

        saved_name = manager.save_template(
            "finance-strict",
            {
                "overall_prompt": "overall",
                "claim_extraction_prompt": "extract",
                "claim_verification_prompt": "verify"
            }
        )
        loaded = manager.load_template(saved_name)
        print("Loaded template:", loaded)
        if loaded["overall_prompt"] == "overall" and "finance-strict" in manager.list_templates():
            print("SUCCESS: Prompt template save/load works.")
        else:
            print("FAILURE: Prompt template save/load mismatch.")

        if default_prompts["overall_prompt"]:
            print("SUCCESS: Default prompt payload is populated.")
        else:
            print("FAILURE: Default prompt payload is empty.")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_sample_dataset():
    print("\nTesting bundled sample dataset...")
    sample_path = os.path.join("data", "sample_eval_dataset.json")

    with open(sample_path, "r", encoding="utf-8") as file:
        sample_dataset = json.load(file)

    temp_dir = tempfile.mkdtemp(prefix="sample_dataset_", dir="data")
    dataset_path = os.path.join(temp_dir, "test_set.json")
    try:
        manager = TestSetManager(data_path=dataset_path)
        manager.import_dataset(sample_dataset)
        loaded = manager.get_dataset()
        print("Loaded sample dataset:", loaded["dataset_name"], len(loaded["samples"]))

        if loaded["dataset_name"] == "finance_hallucination_demo" and len(loaded["samples"]) == 2:
            print("SUCCESS: Bundled sample dataset imports correctly.")
        else:
            print("FAILURE: Bundled sample dataset mismatch.")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_prompt_manager()
    test_sample_dataset()
