import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data_manager.test_set_manager import TestSetManager
from eval_engine.hallucination_evaluator import HallucinationEvaluator
from langchain_core.documents import Document
from rag_engine.financial_rag import FinancialRAG


def test_rag_and_eval():
    print("Testing RAG and evaluation workflow with mocks...")

    print("\n[1] Setting up mock vector store...")
    mock_vector_store = MagicMock()
    mock_retriever = MagicMock()
    mock_docs = [
        Document(page_content="Apple Inc. reported Q3 revenue of $81.4 billion."),
        Document(page_content="Microsoft Azure revenue grew 51% year-over-year.")
    ]
    mock_retriever.invoke.return_value = mock_docs
    mock_vector_store.as_retriever.return_value = mock_retriever

    print("[2] Testing RAG retrieval interface...")
    with patch("rag_engine.financial_rag.ChatOpenAI") as MockChatOpenAI:
        mock_llm = MagicMock()
        MockChatOpenAI.return_value = mock_llm

        rag = FinancialRAG(vector_store=mock_vector_store)
        result = rag.retrieve_context("What was Apple's Q3 revenue?")

        print("RAG result:", result)
        if len(result) == 2 and result[0].page_content.startswith("Apple Inc."):
            print("SUCCESS: RAG retrieval works.")
        else:
            print("FAILURE: Unexpected retrieval output.")

    print("\n[3] Testing dataset manager structure...")
    temp_dir = tempfile.mkdtemp(prefix="eval_dataset_", dir="data")
    dataset_path = os.path.join(temp_dir, "test_eval_dataset.json")
    manager = TestSetManager(data_path=dataset_path)
    manager.import_dataset(
        {
            "dataset_name": "mock_eval",
            "kb_version": "test",
            "samples": [
                {
                    "id": 1,
                    "question": "What was Apple's Q3 revenue?",
                    "candidate_answer": "Apple's Q3 revenue was $91.4 billion.",
                    "label": "positive",
                    "source_model": "weak-model",
                    "source_type": "weak_model",
                    "reference_docs": ["Apple Inc. reported Q3 revenue of $81.4 billion."]
                }
            ]
        }
    )
    loaded_dataset = manager.get_dataset()
    print("Dataset:", loaded_dataset)
    if loaded_dataset["samples"][0]["candidate_answer"]:
        print("SUCCESS: Dataset manager stores structured samples.")
    else:
        print("FAILURE: Dataset manager did not persist structured sample.")

    print("\n[4] Testing overall evaluation mode...")
    with patch("eval_engine.hallucination_evaluator.ChatOpenAI"):
        evaluator = HallucinationEvaluator()
        evaluator._invoke_json = MagicMock(
            return_value={
                "verdict": "hallucinated",
                "confidence": 0.93,
                "reason": "The revenue number conflicts with the evidence.",
                "evidence": ["Apple Inc. reported Q3 revenue of $81.4 billion."],
                "unsupported_parts": ["$91.4 billion"]
            }
        )
        results = evaluator.run_batch_eval(loaded_dataset, rag, mode="overall")
        metrics = evaluator.calculate_classification_metrics(results)

        print("Overall results:", results)
        print("Overall metrics:", metrics)
        if results[0]["predicted_label"] == "positive" and metrics["accuracy"] == 1.0:
            print("SUCCESS: Overall evaluation mode works.")
        else:
            print("FAILURE: Overall evaluation mode mismatch.")

    print("\n[5] Testing claim-level evaluation mode...")
    with patch("eval_engine.hallucination_evaluator.ChatOpenAI"):
        evaluator = HallucinationEvaluator()
        evaluator._invoke_json = MagicMock(
            side_effect=[
                {"claims": ["Apple's Q3 revenue was $91.4 billion."]},
                {
                    "claim": "Apple's Q3 revenue was $91.4 billion.",
                    "verdict": "contradicted",
                    "confidence": 0.95,
                    "reason": "The evidence states $81.4 billion instead.",
                    "evidence": ["Apple Inc. reported Q3 revenue of $81.4 billion."]
                }
            ]
        )
        results = evaluator.run_batch_eval(loaded_dataset, rag, mode="claim")
        metrics = evaluator.calculate_classification_metrics(results)

        print("Claim results:", results)
        print("Claim metrics:", metrics)
        if results[0]["verdict"] == "hallucinated" and metrics["accuracy"] == 1.0:
            print("SUCCESS: Claim-level evaluation mode works.")
        else:
            print("FAILURE: Claim-level evaluation mode mismatch.")

    if os.path.exists(temp_dir):
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_rag_and_eval()
