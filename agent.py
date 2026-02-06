"""LangGraph agent with Schema RAG (Voyage + Atlas Vector Search). Query-example RAG is disabled for speed."""
from typing import Annotated, Dict, List, Literal
from typing_extensions import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from config import get_database, ANTHROPIC_API_KEY
from tools import get_all_tools
from rag import retrieve_schema_context


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


BASE_SYSTEM = """You are a MongoDB expert. You help users query the database named "inferyx" by understanding their natural language prompt and using the following tools:

1. list_collections - Call this first to see which collections exist.
2. get_collection_schema - Call this to see field names and types for one or more collections. For questions that need a JOIN between two collections, get schema for both collections to identify the local and foreign key fields for $lookup.
3. execute_find - Run a simple find query on a single collection (filter and optional projection). Use when the user wants to list or filter documents from one collection.
4. execute_aggregation - Run an aggregation pipeline. Use for: grouping, counting, sorting, or JOINing two collections with $lookup. For a join, use a stage like: {{"$lookup": {{"from": "other_collection", "localField": "field_in_this_collection", "foreignField": "_id", "as": "joined_docs"}}}}.

Always use the tools to answer. Use any relevant schema provided below to prioritize collections and query shape. Then call tools as needed. For "join" or "combine data from two collections", use execute_aggregation with a $lookup stage. Return the final tool result as the answer to the user."""


def _get_last_user_text(messages: List[BaseMessage]) -> str:
    """Return the content of the most recent HumanMessage."""
    for m in reversed(messages):
        if isinstance(m, HumanMessage) and hasattr(m, "content") and m.content:
            return m.content if isinstance(m.content, str) else str(m.content)
    return ""


def _create_agent(db):
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        api_key=ANTHROPIC_API_KEY or None,
        temperature=0,
    )
    tools = get_all_tools(db)
    llm_with_tools = llm.bind_tools(tools)

    # System message is dynamic: base + RAG context (schema + query examples)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "{{system}}"),
        MessagesPlaceholder(variable_name="messages"),
    ])

    chain = prompt | llm_with_tools

    def agent_node(state: AgentState) -> Dict[str, List[BaseMessage]]:
        messages = state["messages"]
        user_text = _get_last_user_text(messages)
        schema_ctx = retrieve_schema_context(user_text, top_k=8) if user_text else ""
        system_parts = [BASE_SYSTEM]
        if schema_ctx:
            system_parts.append("\n\n" + schema_ctx)
        full_system = "".join(system_parts)
        response = chain.invoke({"system": full_system, "messages": messages})
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    def route_after_agent(state: AgentState) -> Literal["tools", "__end__"]:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "__end__"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "__end__": END})
    graph.add_edge("tools", "agent")

    return graph.compile()


def run_agent(user_prompt: str) -> tuple[str, List[BaseMessage]]:
    """Run the agent with the given prompt. Returns (final_answer_text, list of messages)."""
    db = get_database()
    app = _create_agent(db)
    initial = {"messages": [HumanMessage(content=user_prompt)]}
    final_state = None
    for chunk in app.stream(initial, stream_mode="values"):
        final_state = chunk
    messages = (final_state or {}).get("messages", [])
    final_text = "No response generated."
    for m in reversed(messages):
        if isinstance(m, AIMessage) and (not getattr(m, "tool_calls", None) or not m.tool_calls):
            final_text = m.content or final_text
            break
    return final_text, messages
