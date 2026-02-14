import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import traceback

# Setup environment (replace with dummy key if needed, but user has real key)
# Assuming user will run this and has key in env or we can ask them to set it.
# For now, I will use a placeholder and catch the error if it's auth related vs parameter related.
if "OPENAI_API_KEY" not in os.environ:
    print("Warning: OPENAI_API_KEY not found in environment. Please set it.")
    # We can't really test without a key.
    # But let's assume the user runs it.

base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
embed_model_name = "text-embedding-v1"
chat_model_name = "qwen-plus"

print(f"Testing Embeddings with model: {embed_model_name}")
try:
    embeddings = OpenAIEmbeddings(
        openai_api_base=base_url,
        model=embed_model_name,
        check_embedding_ctx_length=False  # Try disabling this first
    )
    text = "这是一个测试文本"
    res = embeddings.embed_query(text)
    print(f"Embeddings success! Dimension: {len(res)}")
except Exception as e:
    print("Embeddings failed:")
    traceback.print_exc()

print("\nTesting Chat with model: {chat_model_name}")
try:
    chat = ChatOpenAI(
        model_name=chat_model_name,
        openai_api_base=base_url,
        temperature=0
    )
    res = chat.invoke("你好")
    print(f"Chat success! Response: {res.content}")
except Exception as e:
    print("Chat failed:")
    traceback.print_exc()
