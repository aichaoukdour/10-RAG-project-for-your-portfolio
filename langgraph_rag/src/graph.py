from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from .nodes import (
    generate_query_or_respond, 
    grade_documents, 
    rewrite_question, 
    generate_answer
)
from .tools import retriever_tool

def get_graph():
    """
    Compiles and returns the LangGraph workflow.
    """
    workflow = StateGraph(MessagesState)

    # 1. Add nodes
    workflow.add_node("agent", generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node("rewrite", rewrite_question)
    workflow.add_node("generate", generate_answer)

    # 2. Define edges
    # Start at the agent
    workflow.add_edge(START, "agent")

    # Agent decides: use tools or end
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "retrieve",
            END: END,
        },
    )

    # After retrieval, grade the documents
    workflow.add_conditional_edges(
        "retrieve",
        grade_documents,
        {
            "generate_answer": "generate",
            "rewrite_question": "rewrite",
        }
    )

    # If rewriting, go back to the agent to try a new search
    workflow.add_edge("rewrite", "agent")

    # If generating, end the workflow
    workflow.add_edge("generate", END)

    # 3. Compile
    return workflow.compile()
