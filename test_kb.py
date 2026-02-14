import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from knowledge_base.document_loader import DocumentLoader
from knowledge_base.vector_store_manager import VectorStoreManager
from langchain_community.embeddings import FakeEmbeddings

def test_kb():
    print("Testing Knowledge Base Module...")
    
    # 1. Test DocumentLoader
    loader = DocumentLoader()
    file_path = os.path.join("data", "test_doc.txt")
    
    print(f"Loading {file_path}...")
    content = loader.load_file(file_path)
    print(f"Loaded content (length {len(content)}):")
    print("-" * 20)
    print(content)
    print("-" * 20)
    
    # Check cleaning
    if "Page 1" in content:
        print("WARNING: Page number not cleaned.")
    else:
        print("SUCCESS: Page number cleaned.")
        
    if "Disclaimer" in content:
        print("WARNING: Disclaimer not cleaned.")
    else:
        print("SUCCESS: Disclaimer cleaned.")

    # 2. Test VectorStoreManager
    print("\nTesting VectorStoreManager...")
    # Use FakeEmbeddings to avoid API Key issues during test
    fake_embeddings = FakeEmbeddings(size=1536)
    vsm = VectorStoreManager(embedding_model=fake_embeddings, persist_directory="./data/chroma_test")
    
    print("Splitting text...")
    docs = vsm.text_splitter(content, chunk_size=50, chunk_overlap=10)
    print(f"Created {len(docs)} chunks.")
    
    print("Adding to vector store...")
    vsm.add_documents(docs)
    
    print("Searching...")
    results = vsm.similarity_search("tech sector", top_k=1)
    if results:
        print(f"Found result: {results[0].page_content}")
        print("SUCCESS: Vector store search works.")
    else:
        print("FAILURE: No results found.")

if __name__ == "__main__":
    test_kb()
