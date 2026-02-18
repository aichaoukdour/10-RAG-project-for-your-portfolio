"""
RAG Chain: Single-call retrieval chain.

Always retrieves context for every query — no routing decision.
This is the "fast path" from the LangChain tutorial: one retrieval
step plus one LLM call, implemented with LangChain's LCEL
(LangChain Expression Language) RunnableSequence.
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline as hf_pipeline

from src.config import LLM_MODEL_NAME, LLM_MAX_NEW_TOKENS, TOP_K


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

RAG_PROMPT = PromptTemplate.from_template(
    "You are a helpful assistant. Use the following context to answer "
    "the question.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_docs(docs):
    """Join retrieved document contents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


def _get_langchain_llm():
    """Wrap the HuggingFace pipeline in a LangChain-compatible LLM."""
    pipe = hf_pipeline(
        "text2text-generation",
        model=LLM_MODEL_NAME,
        max_new_tokens=LLM_MAX_NEW_TOKENS,
    )
    return HuggingFacePipeline(pipeline=pipe)


# ---------------------------------------------------------------------------
# Chain Builder
# ---------------------------------------------------------------------------

def build_rag_chain(vector_store):
    """
    Build a LangChain LCEL chain:
        retriever → format docs → prompt → LLM → parse output

    This mirrors the RAG chain from the tutorial but uses a local
    HuggingFace model instead of an API-based chat model.
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})
    llm = _get_langchain_llm()

    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain


def run_chain(chain, query: str):
    """Invoke the RAG chain and return the answer."""
    print(f"⛓️  Running RAG chain for: '{query}'")
    answer = chain.invoke(query)
    return answer
