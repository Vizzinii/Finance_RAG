import json
import os
import re
from typing import Dict, List

from eval_engine.hallucination_evaluator import HallucinationEvaluator


class PromptTemplateManager:
    DEFAULT_TEMPLATE_NAME = "default"

    def __init__(self, template_dir: str = "./data/prompt_templates"):
        self.template_dir = template_dir
        os.makedirs(self.template_dir, exist_ok=True)

    def _sanitize_name(self, name: str) -> str:
        sanitized = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip())
        return sanitized.strip("_") or self.DEFAULT_TEMPLATE_NAME

    def _template_path(self, name: str) -> str:
        return os.path.join(self.template_dir, f"{self._sanitize_name(name)}.json")

    def get_default_prompts(self) -> Dict[str, str]:
        return {
            "overall_prompt": HallucinationEvaluator.DEFAULT_OVERALL_PROMPT.strip(),
            "claim_extraction_prompt": HallucinationEvaluator.DEFAULT_CLAIM_EXTRACTION_PROMPT.strip(),
            "claim_verification_prompt": HallucinationEvaluator.DEFAULT_CLAIM_VERIFICATION_PROMPT.strip()
        }

    def list_templates(self) -> List[str]:
        saved_templates = []
        for file_name in os.listdir(self.template_dir):
            if file_name.endswith(".json"):
                saved_templates.append(os.path.splitext(file_name)[0])

        names = [self.DEFAULT_TEMPLATE_NAME]
        names.extend(sorted(name for name in saved_templates if name != self.DEFAULT_TEMPLATE_NAME))
        return names

    def save_template(self, name: str, prompts: Dict[str, str]) -> str:
        template_name = self._sanitize_name(name)
        payload = {
            "name": template_name,
            "overall_prompt": prompts.get("overall_prompt", "").strip(),
            "claim_extraction_prompt": prompts.get("claim_extraction_prompt", "").strip(),
            "claim_verification_prompt": prompts.get("claim_verification_prompt", "").strip()
        }
        with open(self._template_path(template_name), "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        return template_name

    def load_template(self, name: str) -> Dict[str, str]:
        template_name = self._sanitize_name(name)
        if template_name == self.DEFAULT_TEMPLATE_NAME:
            return self.get_default_prompts()

        template_path = self._template_path(template_name)
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Prompt template '{template_name}' not found.")

        with open(template_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        return {
            "overall_prompt": data.get("overall_prompt", "").strip(),
            "claim_extraction_prompt": data.get("claim_extraction_prompt", "").strip(),
            "claim_verification_prompt": data.get("claim_verification_prompt", "").strip()
        }
