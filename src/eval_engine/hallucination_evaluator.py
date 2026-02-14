from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

class HallucinationEvaluator:
    def __init__(self, model_name: str = "gpt-3.5-turbo", base_url: str = None, timeout: int = 60, api_key: str = None):
        self.judge_model = ChatOpenAI(
            model_name=model_name, 
            temperature=0,
            base_url=base_url,
            timeout=timeout,
            api_key=api_key
        )
        self.metrics = ['faithfulness', 'answer_relevance']

    def eval_faithfulness(self, answer: str, context: str) -> float:
        """
        Evaluate if the answer is faithful to the context.
        Returns a score between 0.0 and 1.0.
        """
        prompt = PromptTemplate.from_template(
            """
            You are a judge. Evaluate whether the following answer is faithful to the context.
            Answer is faithful if all claims in the answer can be inferred from the context.
            
            Context: {context}
            Answer: {answer}
            
            Respond with a score from 0.0 to 1.0, where 1.0 is fully faithful and 0.0 is hallucination.
            Only return the numeric score.
            """
        )
        chain = prompt | self.judge_model | StrOutputParser()
        try:
            score = chain.invoke({"context": context, "answer": answer})
            return float(score.strip())
        except:
            return 0.0

    def eval_relevance(self, answer: str, question: str) -> float:
        """
        Evaluate if the answer is relevant to the question.
        Returns a score between 0.0 and 1.0.
        """
        prompt = PromptTemplate.from_template(
            """
            You are a judge. Evaluate whether the answer addresses the question.
            
            Question: {question}
            Answer: {answer}
            
            Respond with a score from 0.0 to 1.0, where 1.0 is fully relevant.
            Only return the numeric score.
            """
        )
        chain = prompt | self.judge_model | StrOutputParser()
        try:
            score = chain.invoke({"question": question, "answer": answer})
            return float(score.strip())
        except:
            return 0.0

    def run_batch_eval(self, dataset: List[Dict], rag_engine) -> List[Dict]:
        """
        Run evaluation on a dataset using the provided RAG engine.
        """
        results = []
        for case in dataset:
            question = case['question']
            
            # Generate answer using RAG engine
            rag_result = rag_engine.generate_answer(question)
            answer = rag_result['answer']
            context_docs = rag_result['source_documents']
            context_str = "\n".join(context_docs)
            
            # Evaluate
            faithfulness_score = self.eval_faithfulness(answer, context_str)
            relevance_score = self.eval_relevance(answer, question)
            
            result = {
                "id": case.get('id'),
                "question": question,
                "answer": answer,
                "faithfulness": faithfulness_score,
                "relevance": relevance_score,
                "ground_truth": case.get('ground_truth', '')
            }
            results.append(result)
            
        return results

    def calculate_score(self, results: List[Dict]) -> Dict[str, float]:
        """Calculate average scores."""
        if not results:
            return {"faithfulness": 0.0, "relevance": 0.0}
            
        avg_faithfulness = sum(r['faithfulness'] for r in results) / len(results)
        avg_relevance = sum(r['relevance'] for r in results) / len(results)
        
        return {
            "faithfulness": avg_faithfulness,
            "relevance": avg_relevance
        }
