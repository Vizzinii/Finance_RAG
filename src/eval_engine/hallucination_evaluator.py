import json
from typing import Any, Dict, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


class HallucinationEvaluator:
    DEFAULT_OVERALL_PROMPT = """
    You are a financial fact-checking assistant. Judge whether the candidate answer is supported by the retrieved evidence.

    Rules:
    1. Only use the retrieved evidence. Do not rely on outside knowledge.
    2. If the key facts in the answer are clearly supported by the evidence, verdict must be "supported".
    3. If the key facts in the answer conflict with the evidence, verdict must be "hallucinated".
    4. If the evidence is insufficient to confirm or reject the answer, verdict must be "uncertain".
    5. Return strict JSON only.

    Question:
    {question}

    Candidate Answer:
    {candidate_answer}

    Retrieved Evidence:
    {context}

    Return JSON:
    {{
      "verdict": "supported | hallucinated | uncertain",
      "confidence": 0.0,
      "reason": "short explanation",
      "evidence": ["evidence snippet 1", "evidence snippet 2"],
      "unsupported_parts": ["part that is unsupported or contradicted"]
    }}
    """

    DEFAULT_CLAIM_EXTRACTION_PROMPT = """
    You are a financial text analysis assistant. Break the candidate answer into minimal verifiable factual claims.

    Rules:
    1. Extract only factual claims that can be checked against evidence.
    2. Keep each claim atomic whenever possible.
    3. Return strict JSON only.

    Question:
    {question}

    Candidate Answer:
    {candidate_answer}

    Return JSON:
    {{
      "claims": ["claim 1", "claim 2"]
    }}
    """

    DEFAULT_CLAIM_VERIFICATION_PROMPT = """
    You are a financial fact-checking assistant. Verify the claim using only the retrieved evidence.

    Rules:
    1. Only use the retrieved evidence. Do not rely on outside knowledge.
    2. If the claim is clearly supported by the evidence, verdict must be "supported".
    3. If the claim conflicts with the evidence, verdict must be "contradicted".
    4. If the evidence is insufficient, verdict must be "insufficient_evidence".
    5. Return strict JSON only.

    Question:
    {question}

    Claim:
    {claim}

    Retrieved Evidence:
    {context}

    Return JSON:
    {{
      "claim": "{claim}",
      "verdict": "supported | contradicted | insufficient_evidence",
      "confidence": 0.0,
      "reason": "short explanation",
      "evidence": ["evidence snippet 1", "evidence snippet 2"]
    }}
    """

    def __init__(
        self,
        model_name: str = None,
        base_url: str = None,
        timeout: int = 60,
        api_key: str = None,
        overall_prompt: str = None,
        claim_extraction_prompt: str = None,
        claim_verification_prompt: str = None,
        retrieval_top_k: int = 3
    ):
        self.judge_model = ChatOpenAI(
            model_name=model_name,
            temperature=0,
            base_url=base_url,
            timeout=timeout,
            api_key=api_key
        )
        self.retrieval_top_k = retrieval_top_k
        self.overall_prompt = PromptTemplate.from_template(
            overall_prompt or self.DEFAULT_OVERALL_PROMPT
        )
        self.claim_extraction_prompt = PromptTemplate.from_template(
            claim_extraction_prompt or self.DEFAULT_CLAIM_EXTRACTION_PROMPT
        )
        self.claim_verification_prompt = PromptTemplate.from_template(
            claim_verification_prompt or self.DEFAULT_CLAIM_VERIFICATION_PROMPT
        )

    def _invoke_json(self, prompt: PromptTemplate, variables: Dict[str, Any], fallback: Dict[str, Any]):
        chain = prompt | self.judge_model | StrOutputParser()
        try:
            raw_output = chain.invoke(variables).strip()
            return json.loads(raw_output)
        except Exception:
            return fallback

    def _docs_to_strings(self, docs: List[Any]) -> List[str]:
        snippets = []
        for doc in docs:
            if hasattr(doc, "page_content"):
                snippets.append(doc.page_content)
            elif isinstance(doc, str):
                snippets.append(doc)
        return snippets

    def _predict_label_from_verdict(self, verdict: str) -> str:
        if verdict == "hallucinated":
            return "positive"
        if verdict == "supported":
            return "negative"
        return "uncertain"

    def _retrieve_evidence(self, rag_engine, query: str) -> List[str]:
        if hasattr(rag_engine, "retrieve_context"):
            docs = rag_engine.retrieve_context(query)
            return self._docs_to_strings(docs)

        if hasattr(rag_engine, "vector_store") and hasattr(rag_engine.vector_store, "similarity_search"):
            docs = rag_engine.vector_store.similarity_search(query, top_k=self.retrieval_top_k)
            return self._docs_to_strings(docs)

        raise AttributeError("The provided engine does not expose a retrieval interface.")

    def evaluate_sample_overall(self, sample: Dict[str, Any], rag_engine) -> Dict[str, Any]:
        evidence_docs = self._retrieve_evidence(rag_engine, sample["question"])
        context = "\n\n".join(evidence_docs) if evidence_docs else "No evidence retrieved."

        fallback = {
            "verdict": "uncertain",
            "confidence": 0.0,
            "reason": "Failed to parse judge output.",
            "evidence": [],
            "unsupported_parts": []
        }
        judgment = self._invoke_json(
            self.overall_prompt,
            {
                "question": sample["question"],
                "candidate_answer": sample["candidate_answer"],
                "context": context
            },
            fallback
        )

        verdict = str(judgment.get("verdict", "uncertain")).strip().lower()
        predicted_label = self._predict_label_from_verdict(verdict)
        return {
            "id": sample.get("id"),
            "mode": "overall",
            "question": sample.get("question", ""),
            "candidate_answer": sample.get("candidate_answer", ""),
            "expected_label": sample.get("label", ""),
            "predicted_label": predicted_label,
            "verdict": verdict,
            "confidence": float(judgment.get("confidence", 0.0) or 0.0),
            "reason": judgment.get("reason", ""),
            "evidence": judgment.get("evidence", []),
            "unsupported_parts": judgment.get("unsupported_parts", []),
            "source_model": sample.get("source_model", ""),
            "source_type": sample.get("source_type", ""),
            "reference_docs": sample.get("reference_docs", []),
            "ground_truth": sample.get("ground_truth", ""),
            "is_correct": predicted_label == sample.get("label", "")
        }

    def extract_claims(self, question: str, candidate_answer: str) -> List[str]:
        fallback = {"claims": []}
        result = self._invoke_json(
            self.claim_extraction_prompt,
            {"question": question, "candidate_answer": candidate_answer},
            fallback
        )
        claims = result.get("claims", [])
        return [claim.strip() for claim in claims if isinstance(claim, str) and claim.strip()]

    def evaluate_claim(self, question: str, claim: str, rag_engine) -> Dict[str, Any]:
        evidence_docs = self._retrieve_evidence(rag_engine, f"{question}\n{claim}")
        context = "\n\n".join(evidence_docs) if evidence_docs else "No evidence retrieved."

        fallback = {
            "claim": claim,
            "verdict": "insufficient_evidence",
            "confidence": 0.0,
            "reason": "Failed to parse judge output.",
            "evidence": []
        }
        result = self._invoke_json(
            self.claim_verification_prompt,
            {
                "question": question,
                "claim": claim,
                "context": context
            },
            fallback
        )
        result["claim"] = result.get("claim", claim)
        result["verdict"] = str(result.get("verdict", "insufficient_evidence")).strip().lower()
        result["confidence"] = float(result.get("confidence", 0.0) or 0.0)
        result["evidence"] = result.get("evidence", [])
        result["reason"] = result.get("reason", "")
        return result

    def aggregate_claim_results(self, claim_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not claim_results:
            return {
                "verdict": "uncertain",
                "predicted_label": "uncertain",
                "reason": "No verifiable claims were extracted from the candidate answer.",
                "claim_counts": {
                    "supported": 0,
                    "contradicted": 0,
                    "insufficient_evidence": 0
                }
            }

        contradicted = [item for item in claim_results if item["verdict"] == "contradicted"]
        insufficient = [
            item for item in claim_results if item["verdict"] == "insufficient_evidence"
        ]
        supported = [item for item in claim_results if item["verdict"] == "supported"]

        if contradicted:
            verdict = "hallucinated"
            reason = "At least one factual claim conflicts with the retrieved evidence."
        elif insufficient:
            verdict = "uncertain"
            reason = "Some claims could not be verified from the retrieved evidence."
        else:
            verdict = "supported"
            reason = "All extracted claims are supported by the retrieved evidence."

        return {
            "verdict": verdict,
            "predicted_label": self._predict_label_from_verdict(verdict),
            "reason": reason,
            "claim_counts": {
                "supported": len(supported),
                "contradicted": len(contradicted),
                "insufficient_evidence": len(insufficient)
            }
        }

    def evaluate_sample_claim_level(self, sample: Dict[str, Any], rag_engine) -> Dict[str, Any]:
        claims = self.extract_claims(sample["question"], sample["candidate_answer"])
        claim_results = [
            self.evaluate_claim(sample["question"], claim, rag_engine)
            for claim in claims
        ]
        aggregate = self.aggregate_claim_results(claim_results)
        contradicted_claims = [
            item for item in claim_results if item["verdict"] == "contradicted"
        ]

        return {
            "id": sample.get("id"),
            "mode": "claim",
            "question": sample.get("question", ""),
            "candidate_answer": sample.get("candidate_answer", ""),
            "expected_label": sample.get("label", ""),
            "predicted_label": aggregate["predicted_label"],
            "verdict": aggregate["verdict"],
            "confidence": 0.0,
            "reason": aggregate["reason"],
            "claim_results": claim_results,
            "claim_counts": aggregate["claim_counts"],
            "hallucinated_claims": contradicted_claims,
            "source_model": sample.get("source_model", ""),
            "source_type": sample.get("source_type", ""),
            "reference_docs": sample.get("reference_docs", []),
            "ground_truth": sample.get("ground_truth", ""),
            "is_correct": aggregate["predicted_label"] == sample.get("label", "")
        }

    def run_batch_eval(self, dataset: Any, rag_engine, mode: str = "overall") -> List[Dict[str, Any]]:
        """Run evaluation on a dataset using the provided retrieval-enabled engine."""
        samples = dataset.get("samples", []) if isinstance(dataset, dict) else dataset
        results = []
        for sample in samples:
            if mode == "claim":
                results.append(self.evaluate_sample_claim_level(sample, rag_engine))
            else:
                results.append(self.evaluate_sample_overall(sample, rag_engine))
        return results

    def calculate_classification_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        if not results:
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "uncertain_rate": 0.0
            }

        tp = fp = tn = fn = uncertain = 0
        for result in results:
            expected = result.get("expected_label")
            predicted = result.get("predicted_label")

            if predicted == "uncertain":
                uncertain += 1

            if expected == "positive":
                if predicted == "positive":
                    tp += 1
                else:
                    fn += 1
            elif expected == "negative":
                if predicted == "negative":
                    tn += 1
                else:
                    fp += 1

        total = len(results)
        accuracy = sum(1 for result in results if result.get("is_correct")) / total
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "uncertain_rate": uncertain / total
        }

    def calculate_score(self, results: List[Dict]) -> Dict[str, float]:
        """Backward-compatible alias for the new classification metrics."""
        return self.calculate_classification_metrics(results)
