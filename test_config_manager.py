import os
import shutil
import sys
import tempfile

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from config_manager import AppConfigManager


def test_config_manager():
    print("Testing app config manager...")

    temp_dir = tempfile.mkdtemp(prefix="app_config_", dir="data")
    config_path = os.path.join(temp_dir, "app_config.json")

    try:
        manager = AppConfigManager(config_path=config_path)

        loaded = manager.load_config()
        runtime = manager.get_runtime_config(loaded)
        provider_presets = manager.get_provider_presets(loaded)
        print("Loaded config:", loaded)
        if (
            runtime["provider"] == ""
            and runtime["api_key"] == ""
            and provider_presets == {}
            and os.path.exists(config_path)
        ):
            print("SUCCESS: Default config is created and loaded.")
        else:
            print("FAILURE: Default config was not initialized correctly.")

        saved = manager.save_config(
            {
                "runtime": {
                    "provider": "OpenAI",
                    "base_url": "https://api.openai.com/v1",
                    "chat_model_name": "gpt-4o-mini",
                    "embedding_model_name": "text-embedding-3-small",
                    "api_key": "test-secret-key",
                    "vector_store_directory": "./data/custom_chroma_db",
                    "retrieval_top_k": 5
                },
                "provider_presets": {
                    "OpenAI": {
                        "base_url": "https://api.openai.com/v1",
                        "chat_model_name": "gpt-4o-mini",
                        "embedding_model_name": "text-embedding-3-small"
                    }
                }
            }
        )
        saved_runtime = manager.get_runtime_config(saved)
        print("Saved config:", saved)
        if (
            saved_runtime["provider"] == "OpenAI"
            and saved_runtime["retrieval_top_k"] == 5
            and saved_runtime["api_key"] == "test-secret-key"
        ):
            print("SUCCESS: Config save works.")
        else:
            print("FAILURE: Config save mismatch.")

        reloaded = manager.load_config()
        print("Reloaded config:", reloaded)
        if reloaded == saved:
            print("SUCCESS: Config reload matches saved values.")
        else:
            print("FAILURE: Reloaded config differs from saved values.")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_config_manager()
