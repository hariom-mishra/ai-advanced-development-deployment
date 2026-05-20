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

# === LangGraph Error Handling ===


class RobustState(TypedDict):
    messages: Annotated[list, operator.add]
    error: Optional[str]
    retry_count: int
    max_retries: int
    success: bool


def create_robust_agent():
    """Create agent with built-in error handling."""

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def process_with_retry(state: RobustState) -> dict:
        """Process with retry logic built-in."""

        try:
            # Simulate occasional failure
            if random.random() < 0.3 and state["retry_count"] < 2:
                raise Exception("Simulated processing error")

            response = llm.invoke(state["messages"])

            return {"messages": [response], "success": True, "error": None}

        except Exception as e:
            return {
                "error": str(e),
                "retry_count": state["retry_count"] + 1,
                "success": False,
            }

    def should_continue(state: RobustState) -> Literal["retry", "error", "success"]:
        if state["success"]:
            return "success"
        elif state["retry_count"] < state["max_retries"]:
            return "retry"
        else:
            return "error"

    def handle_error(state: RobustState) -> dict:
        return {
            "messages": [
                AIMessage(
                    content=f"I apologize, but I encountered an error: {state['error']}. "
                    "Please try again later."
                )
            ]
        }

    def finalize(state: RobustState) -> dict:
        return state

    # Build graph
    graph = StateGraph(RobustState)

    graph.add_node("process", process_with_retry)
    graph.add_node("handle_error", handle_error)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "process")
    graph.add_conditional_edges(
        "process",
        should_continue,
        {"retry": "process", "error": "handle_error", "success": "finalize"},
    )
    graph.add_edge("handle_error", END)
    graph.add_edge("finalize", END)

    return graph.compile()


def demo_robust_agent():
    """Demonstrate robust agent with error handling."""

    agent = create_robust_agent()

    print("\nRobust Agent Demo:\n")

    for i in range(3):
        result = agent.invoke(
            {
                "messages": [HumanMessage(content="Hello!")],
                "error": None,
                "retry_count": 0,
                "max_retries": 3,
                "success": False,
            }
        )

        status = "✅ Success" if result["success"] else "❌ Failed"
        print(f"Attempt {i+1}: {status}")
        print(f"  Retries used: {result['retry_count']}")
        print(f"  Response: {result['messages'][-1].content[:50]}...")



if __name__ == "__main__":
    # Example usage of the unreliable API call with retry logic
    # try:
    #     result = unreliable_api_call("Hello, World!")
    #     print(result)
    # except Exception as e:
    #     print(f"API call failed after retries: {e}")

    # Run the retry pattern demonstration
    # demo_retry_pattern()

    # Run the circuit breaker demonstration
    # demo_circuit_breaker()
    # Run the fallback chain demonstration
    # demo_fallback_chain()
    # Run the robust agent demonstration
    demo_robust_agent()
