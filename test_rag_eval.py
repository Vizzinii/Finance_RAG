import os
import sys
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from rag_engine.financial_rag import FinancialRAG
from eval_engine.hallucination_evaluator import HallucinationEvaluator
from knowledge_base.vector_store_manager import VectorStoreManager
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

def test_rag_and_eval():
    print("Testing RAG and Eval Modules (with Mocks)...")

    # 1. Mock Vector Store
    print("\n[1] Setting up Mock Vector Store...")
    mock_vector_store = MagicMock()
    mock_retriever = MagicMock()
    
    # Mock retrieved documents
    mock_docs = [
        Document(page_content="Apple Inc. reported Q3 revenue of $81.4 billion."),
        Document(page_content="Microsoft Azure revenue grew 51% year-over-year.")
    ]
    mock_retriever.invoke.return_value = mock_docs
    mock_vector_store.as_retriever.return_value = mock_retriever

    # 2. Mock LLM for RAG
    print("[2] Setting up Mock LLM for RAG...")
    with patch('rag_engine.financial_rag.ChatOpenAI') as MockChatOpenAI:
        mock_llm = MagicMock()
        # Mock the chain invocation result
        # The chain in FinancialRAG returns a string (StrOutputParser)
        # However, we are mocking the LLM, so we need to see how the chain is constructed.
        # Chain: (input) | prompt | llm | parser
        # If we mock LLM, LLM.invoke() should return a message-like object that StrOutputParser can parse,
        # OR we can mock the entire chain. 
        
        # Let's mock the invoke method of the chain's LLM component
        mock_message = AIMessage(content="Apple's Q3 revenue was $81.4 billion.")
        mock_llm.invoke.return_value = mock_message
        mock_llm.return_value = mock_message  # Handle __call__ if used
        MockChatOpenAI.return_value = mock_llm

        # Instantiate RAG
        rag = FinancialRAG(vector_store=mock_vector_store)
        
        # Test generate_answer
        print("Testing RAG generate_answer...")
        query = "What was Apple's Q3 revenue?"
        result = rag.generate_answer(query)
        
        print(f"Query: {result['query']}")
        print(f"Answer: {result['answer']}")
        print(f"Sources: {result['source_documents']}")
        
        if result['answer'] == "Apple's Q3 revenue was $81.4 billion." and len(result['source_documents']) == 2:
            print("SUCCESS: RAG generation works.")
        else:
            print("FAILURE: RAG generation unexpected output.")

    # 3. Test Evaluation
    print("\n[3] Testing Evaluation Engine...")
    with patch('eval_engine.hallucination_evaluator.ChatOpenAI') as MockEvalLLM:
        mock_eval_llm = MagicMock()
        # Mock responses for faithfulness and relevance
        # First call: Faithfulness (score 1.0)
        # Second call: Relevance (score 0.9)
        mock_msg_1 = AIMessage(content="1.0")
        mock_msg_2 = AIMessage(content="0.9")
        
        mock_eval_llm.invoke.side_effect = [mock_msg_1, mock_msg_2]
        mock_eval_llm.side_effect = [mock_msg_1, mock_msg_2] # Handle __call__
        MockEvalLLM.return_value = mock_eval_llm
        
        evaluator = HallucinationEvaluator()
        
        # Create a dummy dataset
        dataset = [{
            "id": 1,
            "question": "What was Apple's Q3 revenue?",
            "ground_truth": "$81.4 billion"
        }]
        
        # We need to mock the rag_engine passed to run_batch_eval
        # or reuse the one we created if we can patch its generation again.
        # Let's just mock the rag_engine object passed to run_batch_eval
        mock_rag_engine = MagicMock()
        mock_rag_engine.generate_answer.return_value = {
            "answer": "Apple's Q3 revenue was $81.4 billion.",
            "source_documents": ["Apple Inc. reported Q3 revenue of $81.4 billion."]
        }
        
        print("Running batch evaluation...")
        eval_results = evaluator.run_batch_eval(dataset, mock_rag_engine)
        
        print("Eval Results:", eval_results)
        
        scores = evaluator.calculate_score(eval_results)
        print("Scores:", scores)
        
        if scores['faithfulness'] == 1.0 and scores['relevance'] == 0.9:
            print("SUCCESS: Evaluation engine works.")
        else:
            print("FAILURE: Evaluation scores mismatch.")

if __name__ == "__main__":
    test_rag_and_eval()
