import traceback

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.config_manager import AppConfigManager


config_manager = AppConfigManager("./config/app_config.json")
runtime_config = config_manager.get_runtime_config()

api_key = runtime_config.get("api_key", "")
base_url = runtime_config.get("base_url", "")
embed_model_name = runtime_config.get("embedding_model_name", "")
chat_model_name = runtime_config.get("chat_model_name", "")

if not api_key:
    print("Warning: api_key is empty in config/app_config.json")

print(f"Testing Embeddings with model: {embed_model_name}")
try:
    embeddings = OpenAIEmbeddings(
        openai_api_base=base_url,
        model=embed_model_name,
        openai_api_key=api_key,
        check_embedding_ctx_length=False
    )
    text = "This is a configuration smoke test."
    res = embeddings.embed_query(text)
    print(f"Embeddings success! Dimension: {len(res)}")
except Exception:
    print("Embeddings failed:")
    traceback.print_exc()

print(f"\nTesting Chat with model: {chat_model_name}")
try:
    chat = ChatOpenAI(
        model_name=chat_model_name,
        openai_api_base=base_url,
        api_key=api_key,
        temperature=0
    )
    res = chat.invoke("Hello")
    print(f"Chat success! Response: {res.content}")
except Exception:
    print("Chat failed:")
    traceback.print_exc()
