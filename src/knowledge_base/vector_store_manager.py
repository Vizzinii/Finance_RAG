import os
from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

class VectorStoreManager:
    def __init__(self, persist_directory: str = "./data/chroma_db", embedding_model=None, base_url: str = None, model_name: str = "text-embedding-ada-002", api_key: str = None):
        self.persist_directory = persist_directory
        # Ensure the directory exists
        os.makedirs(persist_directory, exist_ok=True)
        
        # Use provided embedding model or default to OpenAIEmbeddings
        # Note: OpenAIEmbeddings requires OPENAI_API_KEY environment variable or api_key param
        if embedding_model:
            self.embedding_model = embedding_model
        else:
            # If base_url is provided, pass it to OpenAIEmbeddings
            self.embedding_model = OpenAIEmbeddings(
                openai_api_base=base_url if base_url else None,
                model=model_name,
                check_embedding_ctx_length=False,  # Disable token counting for compatible APIs
                openai_api_key=api_key,
                chunk_size=10  # Limit batch size for DashScope compatibility (max 25)
            )
        self.collection = None

    def text_splitter(self, text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
        """Split text into chunks."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""]
        )
        # Create documents with metadata placeholder
        return splitter.create_documents([text])

    def add_documents(self, documents: List[Document]):
        """Vectorize and store documents."""
        if not documents:
            return
        
        # Filter out documents with empty content
        valid_docs = [doc for doc in documents if doc.page_content and doc.page_content.strip()]
        if not valid_docs:
            print("No valid documents to add.")
            return

        # Initialize or update Chroma collection
        self.collection = Chroma.from_documents(
            documents=valid_docs,
            embedding=self.embedding_model,
            persist_directory=self.persist_directory
        )
        print(f"Added {len(valid_docs)} documents to vector store.")

    def get_vector_store(self):
        """Get the vector store instance, loading from disk if necessary."""
        if self.collection is None:
            self.collection = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_model
            )
        return self.collection

    def similarity_search(self, query: str, top_k: int = 3) -> List[Document]:
        """Retrieve relevant document chunks."""
        if not query:
            return []
        try:
            store = self.get_vector_store()
            return store.similarity_search(query, k=top_k)
        except Exception as e:
            print(f"Error in similarity_search with query '{query}': {e}")
            raise e

    def as_retriever(self, search_type="similarity", search_kwargs: dict = None):
        """Expose retriever interface for LangChain integration."""
        store = self.get_vector_store()
        return store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
