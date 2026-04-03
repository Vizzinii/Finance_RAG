import json
import os
from copy import deepcopy
from typing import Dict


class AppConfigManager:
    def __init__(self, config_path: str = "./config/app_config.json"):
        self.config_path = config_path

    def get_empty_config(self) -> Dict[str, object]:
        return {
            "runtime": {
                "provider": "",
                "base_url": "",
                "chat_model_name": "",
                "embedding_model_name": "",
                "api_key": "",
                "vector_store_directory": "",
                "retrieval_top_k": 3
            },
            "provider_presets": {}
        }

    def _normalize_provider_presets(self, provider_presets: object) -> Dict[str, Dict[str, str]]:
        if not isinstance(provider_presets, dict):
            return {}

        normalized = {}
        for provider_name, preset in provider_presets.items():
            if not isinstance(preset, dict):
                continue
            normalized[str(provider_name).strip()] = {
                "base_url": str(preset.get("base_url", "") or "").strip(),
                "chat_model_name": str(preset.get("chat_model_name", "") or "").strip(),
                "embedding_model_name": str(preset.get("embedding_model_name", "") or "").strip()
            }
        return normalized

    def _normalize_runtime(
        self,
        runtime: object,
        provider_presets: Dict[str, Dict[str, str]]
    ) -> Dict[str, object]:
        runtime = runtime if isinstance(runtime, dict) else {}
        provider = str(runtime.get("provider", "") or "").strip()
        preset = provider_presets.get(provider, {})

        try:
            retrieval_top_k = max(1, int(runtime.get("retrieval_top_k", 3)))
        except (TypeError, ValueError):
            retrieval_top_k = 3

        return {
            "provider": provider,
            "base_url": str(runtime.get("base_url", "") or preset.get("base_url", "")).strip(),
            "chat_model_name": str(
                runtime.get("chat_model_name", "") or preset.get("chat_model_name", "")
            ).strip(),
            "embedding_model_name": str(
                runtime.get("embedding_model_name", "") or preset.get("embedding_model_name", "")
            ).strip(),
            "api_key": str(runtime.get("api_key", "") or "").strip(),
            "vector_store_directory": str(runtime.get("vector_store_directory", "") or "").strip(),
            "retrieval_top_k": retrieval_top_k
        }

    def _normalize_config(self, config: object) -> Dict[str, object]:
        empty_config = self.get_empty_config()
        if not isinstance(config, dict):
            return empty_config

        provider_presets = self._normalize_provider_presets(config.get("provider_presets", {}))
        runtime = self._normalize_runtime(config.get("runtime", {}), provider_presets)
        return {
            "runtime": runtime,
            "provider_presets": provider_presets
        }

    def load_config(self) -> Dict[str, object]:
        if not os.path.exists(self.config_path):
            empty_config = self.get_empty_config()
            self.save_config(empty_config)
            return empty_config

        try:
            with open(self.config_path, "r", encoding="utf-8") as file:
                config = json.load(file)
        except (json.JSONDecodeError, OSError):
            config = self.get_empty_config()

        normalized = self._normalize_config(config)
        if normalized != config:
            self.save_config(normalized)
        return normalized

    def save_config(self, config: Dict[str, object]) -> Dict[str, object]:
        normalized = self._normalize_config(config)
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as file:
            json.dump(normalized, file, ensure_ascii=False, indent=2)
        return normalized

    def save_runtime_config(self, runtime_config: Dict[str, object]) -> Dict[str, object]:
        current_config = self.load_config()
        current_config["runtime"] = runtime_config
        return self.save_config(current_config)

    def get_runtime_config(self, config: Dict[str, object] = None) -> Dict[str, object]:
        config = self.load_config() if config is None else self._normalize_config(config)
        return deepcopy(config["runtime"])

    def get_provider_presets(self, config: Dict[str, object] = None) -> Dict[str, Dict[str, str]]:
        config = self.load_config() if config is None else self._normalize_config(config)
        return deepcopy(config["provider_presets"])
