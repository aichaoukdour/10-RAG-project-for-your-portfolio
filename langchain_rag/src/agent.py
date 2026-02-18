"""
RAG Agent: Intent-routing agent with retrieval tool.

Uses keyword-based intent detection to decide whether to search
the knowledge base or answer directly. Adapted from the LangChain
tutorial's agentic RAG approach for local HuggingFace models.
"""

from transformers import pipeline as hf_pipeline
from src.config import LLM_MODEL_NAME, LLM_MAX_NEW_TOKENS, SEARCH_KEYWORDS, TOP_K


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

_llm = None


def get_llm():
    """Load the HuggingFace text-to-text generation pipeline (cached)."""
    global _llm
    if _llm is None:
        print(f"🧠 Loading LLM: {LLM_MODEL_NAME}...")
        _llm = hf_pipeline(
            "text2text-generation",
            model=LLM_MODEL_NAME,
            max_new_tokens=LLM_MAX_NEW_TOKENS,
        )
    return _llm


# ---------------------------------------------------------------------------
# Intent Router
# ---------------------------------------------------------------------------

def agent_router(query: str) -> str:
    """
    Decide whether to 'search' the knowledge base or answer 'direct'.
    Uses keyword-based routing to identify intent.

    In a production system this could be replaced with a classifier
    or a model that supports tool calling.
    """
    q_lower = query.lower()
    if any(kw in q_lower for kw in SEARCH_KEYWORDS):
        return "search"
    return "direct"


# ---------------------------------------------------------------------------
# Retrieval Tool
# ---------------------------------------------------------------------------

def retrieve_context(query, vector_store, k=TOP_K):
    """
    Retrieve the top-k most relevant documents from the vector store.
    Returns (serialized_context, raw_docs).

    Mirrors the @tool(response_format="content_and_artifact") pattern
    from the LangChain tutorial, but called directly since local
    models don't support tool calling.
    """
    retrieved_docs = vector_store.similarity_search(query, k=k)
    serialized = "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


# ---------------------------------------------------------------------------
# Agent Execution
# ---------------------------------------------------------------------------

def run_agent(query, vector_store):
    """
    Execute the RAG agent:
    1. Route the query (search vs. direct).
    2. If search: retrieve context, build an augmented prompt, generate.
    3. If direct: generate an answer without retrieval.
    """
    action = agent_router(query)
    llm = get_llm()

    if action == "search":
        print(f"🕵️  Agent decided to SEARCH for: '{query}'")
        context, docs = retrieve_context(query, vector_store)
        prompt = (
            f"Use the following context to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            f"Answer:"
        )
        print(f"📎 Retrieved {len(docs)} documents.")
    else:
        print(f"🤖 Agent decided to answer DIRECTLY: '{query}'")
        prompt = query

    response = llm(prompt)[0]["generated_text"]
    return response
