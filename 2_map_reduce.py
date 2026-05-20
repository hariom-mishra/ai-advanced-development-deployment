"""
Parallel Agent Execution in LangGraph
Running multiple agents simultaneously
"""

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
import asyncio
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Map-Reduce Pattern
class MapReduceState(TypedDict):
    documents: list[str]
    summaries: list[str]
    final_summary: str


def create_map_reduce_summarizer():
    """Summarize multiple documents in parallel."""

    def map_summarize(state: MapReduceState) -> dict:
        """Summarize each document (runs in parallel for each)."""
        summaries = []
        for doc in state["documents"]:
            response = llm.invoke(
                [
                    SystemMessage(content="Summarize this document in 2-3 sentences."),
                    HumanMessage(content=doc),
                ]
            )
            summaries.append(response.content)
        return {"summaries": summaries}

    def reduce_combine(state: MapReduceState) -> dict:
        """Combine all summaries."""
        all_summaries = "\n\n".join(
            [f"Summary {i+1}: {s}" for i, s in enumerate(state["summaries"])]
        )

        response = llm.invoke(
            [
                SystemMessage(
                    content="Combine these summaries into one coherent overview."
                ),
                HumanMessage(content=all_summaries),
            ]
        )
        return {"final_summary": response.content}

    graph = StateGraph(MapReduceState)
    graph.add_node("map", map_summarize)
    graph.add_node("reduce", reduce_combine)

    graph.add_edge(START, "map")
    graph.add_edge("map", "reduce")
    graph.add_edge("reduce", END)

    return graph.compile()


def demo_map_reduce():
    """Demo map-reduce pattern."""

    agent = create_map_reduce_summarizer()

    documents = [
        "Python is a high-level programming language known for its simplicity and readability. It supports multiple programming paradigms and has a vast ecosystem of libraries.",
        "Machine learning is a subset of AI that enables systems to learn from data. Common approaches include supervised, unsupervised, and reinforcement learning.",
        "Cloud computing provides on-demand access to computing resources. Major providers include AWS, Azure, and Google Cloud Platform.",
    ]

    print("\nMap-Reduce Summarization Demo:\n")

    result = agent.invoke(
        {"documents": documents, "summaries": [], "final_summary": ""}
    )

    print("Individual summaries:")
    for i, summary in enumerate(result["summaries"]):
        print(f"  {i+1}. {summary}")

    print(f"\nCombined summary:\n{result['final_summary']}")


if __name__ == "__main__":
    # demo_parallel_execution()
    demo_map_reduce()