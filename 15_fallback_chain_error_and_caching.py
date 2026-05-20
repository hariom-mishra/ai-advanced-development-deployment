import random
from typing import Literal, Optional, Callable
from functools import wraps
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from typing_extensions import TypedDict, Annotated
import operator
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()


# === Model Fallback Chain ===
class FallbackChain:
    """Try multiple models in order until one succeeds."""

    def __init__(self):
        self.models = [
            ("gpt-4o-mini", ChatOpenAI(model="gpt-4o-mini", temperature=0, timeout=10)),
            ("gpt-4o", ChatOpenAI(model="gpt-4o", temperature=0, timeout=10)),
            (
                "claude-sonnet",
                ChatAnthropic(
                    model="claude-sonnet-4-5-20250929", temperature=0, timeout=10
                ),
            ),
        ]
        self.cache = {}

    @traceable(name="fallback_invoke")
    def invoke(self, query: str, use_cache: bool = True) -> tuple[str, str]:
        """
        Invoke with fallbacks.
        Returns: (response, model_used)
        """

        # Check cache first
        if use_cache and query in self.cache:
            return self.cache[query], "cache"

        errors = []

        for model_name, model in self.models:
            try:
                response = model.invoke(query)
                result = response.content

                # Cache successful response
                self.cache[query] = result

                return result, model_name

            except Exception as e:
                errors.append(f"{model_name}: {str(e)}")
                continue

        # All models failed
        raise Exception(f"All models failed: {errors}")


def demo_fallback_chain():
    """Demonstrate fallback chain."""

    chain = FallbackChain()

    print("\nFallback Chain Demo:\n")

    queries = [
        "What is 2 + 2?",
        "What is Python?",
        "What is 2 + 2?",  # Should hit cache
    ]

    for query in queries:
        try:
            result, model = chain.invoke(query)
            print(f"Query: {query}")
            print(f"  Model: {model}")
            print(f"  Response: {result[:50]}...")
        except Exception as e:
            print(f"Query: {query}")
            print(f"  ❌ Error: {e}")

