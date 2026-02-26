# src/evaluation/evaluator.py
from dataclasses import dataclass
from typing import Optional
import json

@dataclass
class RAGEvalResult:
    query: str
    retrieved_chunks: list[dict]
    generated_answer: str
    
    # Retrieval metrics
    retrieval_precision: float    # fraction of retrieved chunks that are relevant
    retrieval_recall: float       # fraction of relevant chunks that were retrieved
    mrr: float                    # Mean Reciprocal Rank
    
    # Generation metrics
    faithfulness: float           # Is the answer grounded in retrieved context?
    answer_relevance: float       # Does the answer address the question?
    context_utilization: float    # How well did the LLM use the context?
    
    # Legal-specific
    citation_accuracy: float      # Are cited sections correct?
    hallucination_detected: bool  # Did the LLM make up legal provisions?

class VakilBotEvaluator:
    """
    Evaluation suite for VakilBot.
    Uses a golden QA dataset of known Indian law questions + answers.
    """
    
    GOLDEN_DATASET = [
        {
            "query": "What is the penalty for hacking under the IT Act?",
            "expected_act": "Information Technology Act, 2000",
            "expected_sections": ["66"],
            "expected_keywords": ["three years", "imprisonment", "fine", "computer"]
        },
        {
            "query": "How can a woman file a domestic violence complaint?",
            "expected_act": "Protection of Women from Domestic Violence Act, 2005",
            "expected_sections": ["12", "18", "19"],
            "expected_keywords": ["protection officer", "magistrate", "application"]
        },
    ]
    
    def evaluate_retrieval(self, results: list[dict], expected: dict) -> dict:
        """Evaluate retrieval quality against ground truth."""
        retrieved_sections = {r["section_number"] for r in results if r.get("section_number")}
        expected_sections = set(expected["expected_sections"])
        
        # Precision: of what we retrieved, how much was relevant?
        relevant_retrieved = retrieved_sections & expected_sections
        precision = len(relevant_retrieved) / len(retrieved_sections) if retrieved_sections else 0
        
        # Recall: of what should have been retrieved, how much did we get?
        recall = len(relevant_retrieved) / len(expected_sections) if expected_sections else 0
        
        # F1
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # MRR: rank of first relevant result
        mrr = 0.0
        for rank, result in enumerate(results, 1):
            if result.get("section_number") in expected_sections:
                mrr = 1.0 / rank
                break
        
        # Act accuracy
        retrieved_acts = {r["act_name"] for r in results}
        act_hit = expected["expected_act"] in retrieved_acts
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mrr": mrr,
            "act_accuracy": float(act_hit),
            "relevant_sections_retrieved": list(relevant_retrieved)
        }
    
    def evaluate_generation(self, answer: str, context: str, expected: dict) -> dict:
        """
        LLM-as-judge evaluation for answer quality.
        Uses GPT-4 to score faithfulness and relevance.
        """
        from openai import OpenAI
        client = OpenAI()
        
        judge_prompt = f"""Evaluate this legal AI response on three dimensions.
        
Query: {expected['query']}
Retrieved Context: {context[:2000]}
Generated Answer: {answer}

Rate each (0.0-1.0):
1. FAITHFULNESS: Is every claim in the answer supported by the context? (1.0 = fully grounded, 0.0 = hallucinated)
2. RELEVANCE: Does the answer address the query? (1.0 = directly answers, 0.0 = off-topic)  
3. CITATION_ACCURACY: Are the cited act/section numbers present in context? (1.0 = all correct)

Respond ONLY as JSON: {{"faithfulness": 0.0, "relevance": 0.0, "citation_accuracy": 0.0}}"""
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.0,
            max_tokens=100
        )
        
        try:
            scores = json.loads(response.choices[0].message.content)
            return scores
        except:
            return {"faithfulness": 0.0, "relevance": 0.0, "citation_accuracy": 0.0}
    
    def run_benchmark(self, vakilbot) -> dict:
        """Run full evaluation benchmark."""
        all_retrieval_scores = []
        all_generation_scores = []
        
        for example in self.GOLDEN_DATASET:
            # Get retrieval results
            results = vakilbot.searcher.hybrid_search(example["query"], k=5)
            
            # Get generated answer
            answer = vakilbot.answer(example["query"], stream=False)
            context = vakilbot.searcher._build_context(results) if hasattr(vakilbot.searcher, '_build_context') else ""
            
            # Evaluate
            retrieval_scores = self.evaluate_retrieval(results, example)
            generation_scores = self.evaluate_generation(answer, str(results), example)
            
            all_retrieval_scores.append(retrieval_scores)
            all_generation_scores.append(generation_scores)
        
        # Aggregate
        avg_scores = {
            "retrieval": {
                "avg_precision": sum(s["precision"] for s in all_retrieval_scores) / len(all_retrieval_scores),
                "avg_recall": sum(s["recall"] for s in all_retrieval_scores) / len(all_retrieval_scores),
                "avg_f1": sum(s["f1"] for s in all_retrieval_scores) / len(all_retrieval_scores),
                "avg_mrr": sum(s["mrr"] for s in all_retrieval_scores) / len(all_retrieval_scores),
                "avg_act_accuracy": sum(s["act_accuracy"] for s in all_retrieval_scores) / len(all_retrieval_scores),
            },
            "generation": {
                "avg_faithfulness": sum(s["faithfulness"] for s in all_generation_scores) / len(all_generation_scores),
                "avg_relevance": sum(s["relevance"] for s in all_generation_scores) / len(all_generation_scores),
                "avg_citation_accuracy": sum(s["citation_accuracy"] for s in all_generation_scores) / len(all_generation_scores),
            }
        }
        
        return avg_scores
