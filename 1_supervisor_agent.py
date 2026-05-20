from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing_extensions import TypedDict,Annotated
from typing import Literal

load_dotenv()

'''goal: given topic you need to reasearch, write and critic 
on the topic and supervise all of these agent using another
 agent supervisor once task completed finish'''

class SupervisorState(BaseModel):
    messages: Annotated[list[BaseMessage],add_messages]
    next_agent: str = None
    task_completed: bool = False
    final_response: str = None


def create_supervisor_agent():
    """create a supervisor with specialist agents"""
    llm = ChatOpenAI(model="gpt-4.1-mini")

    class RouteDecision(BaseModel):
        next: Literal["researcher", "writer", "critic", "FINISH"] = Field(description ="Next agent to call or finish if task is completed." ),
        reasoning: str = Field(description ="reasoning for decision")
        
    supervisor_llm = llm.with_structured_output(RouteDecision)


    def supervisor(state: SupervisorState) -> dict:
        system_prompt = """You are a supervisor managing a team of specialists:

        1. researcher - Gathers information and facts
        2. writer - Creates content and text
        3. critic - Reviews and improves work

        Based on the conversation, decide which agent should act next.
        If the task is complete, respond with FINISH.

        Current conversation shows the progress so far."""
        
        messages = [SystemMessage(content=system_prompt)] + state.messages

        decision = supervisor_llm.invoke(messages)
        if decision.next =="FINISH":
            return {"next_agent":"FINISH", "task_completed":True}

        return {"next_agent":decision.next,
         "messages": [AIMessage(content=f"[Supervisor] Routing to {decision.next}: {decision.reasoning}")]
         }
    

    def researcher(state: SupervisorState)->dict:
        prompt = ChatPromptTemplate([
            ('system', 'You are a researcher specialist.gather information and facts relevant to the task. Be thorough but concise.'),
            ('human', 'Task context:\n{context}\n\nProvide your research findings.')
        ])

        # get task from first human message
        task = next((m.content for m in state.messages if isinstance(m, HumanMessage)), "")
        response = llm.invoke(prompt.format_messages(context=task))

        return {"messages": [AIMessage(content=f"[Researcher] {response.content}")]}

    def writer(state: SupervisorState) -> dict:
        prompt = ChatPromptTemplate.from_messages([
            ('system', 'You are a writing specialist. Create clear, engaging content based on the available information.'),
            ('human', 'Previous work:\n{context}\n\nWrite the content.')
        ])

        context = "\n".join([m.content for m in state.messages[-5:]])
        response = llm.invoke(prompt.format_messages(context=context))

        return {"messages": [AIMessage(content=f"[Writer] {response.content}")]}

    def critic(state: SupervisorState) -> dict:
        prompt = ChatPromptTemplate.from_messages([
            ('system', 'You are a quality critic. Review the work and provide constructive feedback. If the work is good, say so.'),
            ('human','Work to review:\n{context}\n\nProvide your critique.')
        ])

        context = "\n".join([m.content for m in state.messages[-3:]])
        response = llm.invoke(prompt.format_messages(context=context))

        return {"messages": [AIMessage(content=f"[Critic] {response.content}")]}

    def finalize(state: SupervisorState) -> dict:
        for message in reversed(state.messages):
            if(isinstance(message, AIMessage) and "[Writer]" in message.content):
                content = message.content.replace("[Writer]", "")
                return {"final_response": content}
        return {"final_response": "task completed"}

    # Route based on supervisor decision
    def route_to_agent(state: SupervisorState) -> str:
        if state.task_completed:
            return "finalize"
        return state.next_agent


    graph = StateGraph(SupervisorState)

    #nodes
    graph.add_node("researcher", researcher)
    graph.add_node("writer", writer)
    graph.add_node("critic", critic)
    graph.add_node("finalize", finalize)
    graph.add_node("supervisor", supervisor)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
         route_to_agent, 
         {
            "researcher": "researcher",
            "writer": "writer",
            "critic": "critic",
            "finalize": "finalize"
         }
    )
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("writer", "supervisor")
    graph.add_edge("critic", "supervisor")
    graph.add_edge("finalize", END)

    return graph.compile()

def demo_supervisor():
    agent = create_supervisor_agent()
    print("demo hai ji: ")
    result = agent.invoke({
        "messages": [HumanMessage(content="Write a short blog on benefits of ai")]
    })

    print(result["final_response"])

demo_supervisor()