import hashlib
import json
from typing import Optional, Callable
from functools import lru_cache
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()

# === Model Routing ===


class ModelRouter:
    """Route queries to appropriate model based on complexity."""

    def __init__(self):
        self.cheap_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.expensive_model = ChatOpenAI(model="gpt-4o", temperature=0)
        self.classifier = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def classify_complexity(self, query: str) -> str:
        """Classify query complexity."""

        prompt = ChatPromptTemplate.from_template(
            """
Classify this query's complexity as 'simple' or 'complex'.

Simple: Basic facts, short answers, simple calculations
Complex: Analysis, reasoning, creative tasks, multi-step problems

Query: {query}

Respond with only: simple or complex
"""
        )

        response = self.classifier.invoke(prompt.format(query=query))
        return response.content.strip().lower()

    @traceable(name="routed_query")
    def invoke(self, query: str) -> tuple[str, str, float]:
        """
        Route and invoke query.
        Returns: (response, model_used, estimated_cost)
        """
        complexity = self.classify_complexity(query)

        if complexity == "simple":
            model = self.cheap_model
            model_name = "gpt-4o-mini"
            cost_per_1k = 0.00015  # Input cost
        else:
            model = self.expensive_model
            model_name = "gpt-4o"
            cost_per_1k = 0.0025  # Input cost

        response = model.invoke(query)

        # Estimate cost (rough)
        tokens = len(query.split()) * 1.3  # Rough token estimate
        estimated_cost = (tokens / 1000) * cost_per_1k

        return response.content, model_name, estimated_cost


def demo_model_routing():
    """Demonstrate model routing."""

    router = ModelRouter()

    queries = [
        "What is 2 + 2?",  # Simple
        "Analyze the economic implications of AI on the job market.",  # Complex
        "What color is the sky?",  # Simple
    ]

    print("Model Routing Demo:\n")

    total_cost = 0
    for query in queries:
        result, model, cost = router.invoke(query)
        total_cost += cost
        print(f"Query: {query[:50]}...")
        print(f"  Model: {model}")
        print(f"  Est. Cost: ${cost:.6f}")
        print(f"  Response: {result[:50]}...")

    print(f"\nTotal Estimated Cost: ${total_cost:.6f}")
