from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class FinancialRAG:
    def __init__(self, vector_store, model_name: str = "gpt-3.5-turbo", base_url: str = None, timeout: int = 60, api_key: str = None):
        self.vector_store = vector_store
        # Initialize LLM (requires OPENAI_API_KEY or api_key param)
        self.llm = ChatOpenAI(
            model_name=model_name, 
            temperature=0,
            base_url=base_url,
            timeout=timeout,
            api_key=api_key
        )
        
        # Financial expert prompt
        self.prompt_template = PromptTemplate.from_template(
            """You are a professional financial analyst assistant. 
            Use the following pieces of retrieved context to answer the question. 
            If the context does not contain enough information to answer the question, say that you don't know based on the context.
            Do not make up information. 
            Always cite the source if possible (though context provided here is a merged string).
            
            Context:
            {context}
            
            Question: {question}
            
            Answer:"""
        )
        
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        
        # Build the chain
        self.chain = (
            {"context": self.retriever | self._format_docs, "question": RunnablePassthrough()}
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )

    def _format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def retrieve_context(self, query: str) -> List[Any]:
        """Retrieve raw documents for inspection."""
        return self.retriever.invoke(query)

    def generate_answer(self, query: str) -> Dict[str, Any]:
        """
        Generate answer for the query.
        Returns a dictionary with 'answer' and 'source_documents'.
        """
        # We need to manually run retrieval if we want to return source docs with the answer
        # or use a chain that returns sources.
        # For simplicity, let's do it in two steps to expose sources clearly.
        
        docs = self.retrieve_context(query)
        context_str = self._format_docs(docs)
        
        chain_input = {"context": context_str, "question": query}
        # We can invoke the prompt+llm part directly
        answer_chain = (
            self.prompt_template 
            | self.llm 
            | StrOutputParser()
        )
        
        answer = answer_chain.invoke(chain_input)
        
        return {
            "query": query,
            "answer": answer,
            "source_documents": [doc.page_content for doc in docs] # In real app, include metadata
        }
