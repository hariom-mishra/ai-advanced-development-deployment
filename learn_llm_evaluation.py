"""
Testing & Evaluation Patterns
Building reliable LLM applications
"""

import pytest
from unittest.mock import Mock, patch
from typing import Callable
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langsmith import traceable, Client
from dotenv import load_dotenv

load_dotenv()

# === Evaluation Framework ===
class LLMEvaluator:
    """Use LLM to evaluate LLM outputs."""

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    @traceable(name="evaluate_response")
    def evaluate(self, question: str, response: str, reference: str = None) -> dict:
        """Evaluate a response on multiple dimensions."""

        eval_prompt = ChatPromptTemplate.from_template(
            """
Evaluate this response on a scale of 1-10 for each criterion.

Question: {question}
Response: {response}
{reference_section}

Rate each criterion (1-10):
1. Correctness: Is the information accurate?
2. Relevance: Does it answer the question?
3. Clarity: Is it easy to understand?
4. Completeness: Does it fully address the question?

Respond with ONLY a JSON object:
{{"correctness": X, "relevance": X, "clarity": X, "completeness": X, "overall": X}}
"""
        )

        reference_section = ""
        if reference:
            reference_section = f"Reference answer: {reference}"

        import json

        response_obj = self.llm.invoke(
            eval_prompt.format(
                question=question,
                response=response,
                reference_section=reference_section,
            )
        )

        try:
            scores = json.loads(response_obj.content)
            return scores
        except json.JSONDecodeError:
            return {"error": "Failed to parse evaluation"}


def demo_evaluation():
    """Demonstrate LLM evaluation."""

    evaluator = LLMEvaluator()

    # Test case
    question = "Explain what machine learning is in simple terms."
    response = "Machine learning is when computers learn from data instead of being explicitly programmed. It's like teaching a child by showing examples rather than giving them rules."
    reference = "Machine learning is a type of artificial intelligence where computers learn patterns from data to make predictions or decisions."

    print("LLM Evaluation Demo:\n")
    print(f"Question: {question}")
    print(f"Response: {response}")

    scores = evaluator.evaluate(question, response, reference)

    print("\nScores:")
    for metric, score in scores.items():
        print(f"  {metric}: {score}/10")


# === Regression Testing ===
class RegressionTestRunner:
    """Run regression tests against a test dataset."""

    def __init__(self, chain: Callable):
        self.chain = chain
        self.evaluator = LLMEvaluator()

    @traceable(name="regression_test")
    def run(self, test_cases: list[dict]) -> dict:
        """
        Run regression tests.

        test_cases: [{"input": ..., "expected": ...}, ...]
        """
        results = []
        total_score = 0

        for case in test_cases:
            # Get response from chain
            response = self.chain(case["input"])

            # Evaluate
            scores = self.evaluator.evaluate(
                question=case["input"],
                response=response,
                reference=case.get("expected"),
            )

            overall = scores.get("overall", 0)
            total_score += overall

            results.append(
                {
                    "input": case["input"],
                    "response": response,
                    "expected": case.get("expected"),
                    "scores": scores,
                    "passed": overall >= 7,  # Threshold
                }
            )

        return {
            "total": len(results),
            "passed": sum(1 for r in results if r["passed"]),
            "average_score": total_score / len(results) if results else 0,
            "results": results,
        }


def demo_regression_testing():
    """Demonstrate regression testing."""

    # Simple chain to test
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def qa_chain(question: str) -> str:
        return llm.invoke(question).content

    # Test cases
    test_cases = [
        {
            "input": "What is Python?",
            "expected": "Python is a programming language known for its simplicity.",
        },
        {"input": "What is 10 * 5?", "expected": "50"},
    ]

    runner = RegressionTestRunner(qa_chain)

    print("\nRegression Test Results:\n")

    results = runner.run(test_cases)

    print(f"Passed: {results['passed']}/{results['total']}")
    print(f"Average Score: {results['average_score']:.1f}/10")

    for r in results["results"]:
        status = "✅" if r["passed"] else "❌"
        print(f"\n{status} {r['input']}")
        print(f"   Response: {r['response'][:50]}...")
        print(f"   Overall Score: {r['scores'].get('overall', 'N/A')}/10")
