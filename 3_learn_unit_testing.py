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

# === Unit Testing with Mocks ===
class QAChain:
    """Simple Q&A chain for testing."""

    def __init__(self, llm=None):
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.prompt = ChatPromptTemplate.from_template(
            "Answer this question: {question}"
        )

    def ask(self, question: str) -> str:
        prompt_value = self.prompt.invoke({"question": question})
        response = self.llm.invoke(prompt_value)
        return response.content


def test_qa_chain_with_mock():
    """Test QA chain with mocked LLM."""

    # Create mock LLM
    mock_llm = Mock()
    mock_llm.invoke.return_value = AIMessage(content="Paris")

    # Test with mock
    chain = QAChain(llm=mock_llm)
    result = chain.ask("What is the capital of France?")

    assert result == "Paris"
    mock_llm.invoke.assert_called_once()


def test_qa_chain_handles_empty_response():
    """Test chain handles empty responses."""

    mock_llm = Mock()
    mock_llm.invoke.return_value = AIMessage(content="")

    chain = QAChain(llm=mock_llm)
    result = chain.ask("Empty question")

    assert result == ""
