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

# === Integration Testing with Real LLM ===
class IntegrationTestSuite:
    """Integration tests with real LLM calls."""

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    @traceable(name="integration_test")
    def test_basic_qa(self) -> dict:
        """Test basic question answering."""

        test_cases = [
            {
                "question": "What is 2 + 2?",
                "expected_contains": ["4", "four"],
            },
            {
                "question": "What color is the sky on a clear day?",
                "expected_contains": ["blue"],
            },
        ]

        results = []
        for case in test_cases:
            response = self.llm.invoke(case["question"])
            content = response.content.lower()

            passed = any(exp.lower() in content for exp in case["expected_contains"])

            # "The answer is 4" or "2 + 2 equals four" or "That would be 4."

            results.append(
                {
                    "question": case["question"],
                    "response": response.content,
                    "passed": passed,
                }
            )

        return {
            "total": len(results),
            "passed": sum(1 for r in results if r["passed"]),
            "results": results,
        }


def demo_integration_tests():
    """Run integration tests."""

    suite = IntegrationTestSuite()

    print("Integration Test Results:\n")

    results = suite.test_basic_qa()

    print(f"Passed: {results['passed']}/{results['total']}")

    for r in results["results"]:
        status = "✅" if r["passed"] else "❌"
        print(f"{status} {r['question']}")
        print(f"   Response: {r['response'][:50]}...")

