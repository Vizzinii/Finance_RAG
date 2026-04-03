import os
import shutil
import sys
import tempfile
from unittest.mock import MagicMock, patch

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

    # 1.1 Test DOCX dispatch
    print("\nTesting DOCX loader dispatch...")
    docx_fd, docx_path = tempfile.mkstemp(suffix=".docx", dir="data")
    os.close(docx_fd)
    try:
        with patch.object(loader, "_load_docx", return_value="DOCX content") as mock_docx_loader:
            docx_content = loader.load_file(docx_path)
            if mock_docx_loader.called and "DOCX content" in docx_content:
                print("SUCCESS: DOCX format is supported.")
            else:
                print("FAILURE: DOCX dispatch failed.")
    finally:
        if os.path.exists(docx_path):
            try:
                os.remove(docx_path)
            except PermissionError:
                pass

    # 1.2 Test DOC dispatch
    print("Testing DOC loader dispatch...")
    doc_fd, doc_path = tempfile.mkstemp(suffix=".doc", dir="data")
    os.close(doc_fd)
    try:
        with patch.object(loader, "_load_doc", return_value="DOC content") as mock_doc_loader:
            doc_content = loader.load_file(doc_path)
            if mock_doc_loader.called and "DOC content" in doc_content:
                print("SUCCESS: DOC format is supported.")
            else:
                print("FAILURE: DOC dispatch failed.")
    finally:
        if os.path.exists(doc_path):
            try:
                os.remove(doc_path)
            except PermissionError:
                pass

    # 2. Test VectorStoreManager
    print("\nTesting VectorStoreManager...")
    # Use FakeEmbeddings to avoid API Key issues during test
    fake_embeddings = FakeEmbeddings(size=1536)
    temp_dir = tempfile.mkdtemp(prefix="chroma_test_", dir="data")
    try:
        vsm = VectorStoreManager(embedding_model=fake_embeddings, persist_directory=temp_dir)
        
        print("Splitting text...")
        docs = vsm.text_splitter(content, chunk_size=50, chunk_overlap=10)
        print(f"Created {len(docs)} chunks.")

        mock_collection = MagicMock()
        mock_collection.similarity_search.return_value = [
            MagicMock(page_content="The tech sector is leading the growth.")
        ]

        with patch("knowledge_base.vector_store_manager.Chroma") as MockChroma:
            MockChroma.from_documents.return_value = mock_collection
            MockChroma.return_value = mock_collection

            print("Adding to vector store...")
            vsm.add_documents(docs)
            
            print("Searching...")
            results = vsm.similarity_search("tech sector", top_k=1)
            if results:
                print(f"Found result: {results[0].page_content}")
                print("SUCCESS: Vector store search works.")
            else:
                print("FAILURE: No results found.")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    test_kb()
