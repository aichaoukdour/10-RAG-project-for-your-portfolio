from typing import Literal
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from .config import LLM_MODEL
from .tools import retriever_tool

# Initialize the model
# Using ChatOllama for tool binding and structured output support (if available)
model = ChatOllama(model=LLM_MODEL, temperature=0)

# --- Node 1: Generate Query or Respond ---
def generate_query_or_respond(state: MessagesState):
    """
    Decides whether to retrieve information using a tool or respond directly.
    """
    print("🤖 Node: Generate Query or Respond")
    # Bind tools to the model
    model_with_tools = model.bind_tools([retriever_tool])
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# --- Node 2: Grade Documents ---
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )

def grade_documents(state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
    """
    Determines whether the retrieved documents are relevant to the user question.
    """
    print("🔎 Node: Grade Documents")
    
    question = state["messages"][0].content
    # The last message is the tool output (context)
    last_message = state["messages"][-1]
    
    if hasattr(last_message, "content"):
        context = last_message.content
    else:
        context = str(last_message)

    grade_prompt = (
        "You are a grader assessing relevance of a retrieved document to a user question.\n"
        f"Retrieved Document: \n\n {context} \n\n"
        f"User Question: {question} \n\n"
        "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.\n"
        "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.\n"
        "Respond ONLY with the JSON: {\"binary_score\": \"yes\"} or {\"binary_score\": \"no\"}"
    )

    # Some local models might struggle with with_structured_output, 
    # so we'll use a robust approach or fallback if needed.
    try:
        grader = model.with_structured_output(GradeDocuments)
        response = grader.invoke([{"role": "user", "content": grade_prompt}])
        score = response.binary_score.lower()
    except Exception as e:
        print(f"⚠️ Structured output failed, falling back to simple check: {e}")
        # Simple fallback parsing
        raw_response = model.invoke([{"role": "user", "content": grade_prompt}]).content.lower()
        score = "yes" if "yes" in raw_response else "no"

    print(f"📊 Relevance Score: {score}")
    if score == "yes":
        return "generate_answer"
    else:
        return "rewrite_question"

# --- Node 3: Rewrite Question ---
def rewrite_question(state: MessagesState):
    """
    Rewrites the original question to improve retrieval relevance.
    """
    print("✍️ Node: Rewrite Question")
    
    question = state["messages"][0].content
    rewrite_prompt = (
        "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
        f"Initial question: {question}\n"
        "Formulate an improved, more detailed question for professional blog search."
    )
    
    response = model.invoke([{"role": "user", "content": rewrite_prompt}])
    print(f"🔄 Rewritten Question: {response.content}")
    return {"messages": [HumanMessage(content=response.content)]}

# --- Node 4: Generate Answer ---
def generate_answer(state: MessagesState):
    """
    Generates a final answer based on the retrieved context.
    """
    print("💬 Node: Generate Answer")
    
    question = state["messages"][0].content
    context = state["messages"][-1].content
    
    gen_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "Use three sentences maximum and keep the answer concise.\n"
        f"Question: {question}\n"
        f"Context: {context}"
    )
    
    response = model.invoke([{"role": "user", "content": gen_prompt}])
    return {"messages": [response]}
