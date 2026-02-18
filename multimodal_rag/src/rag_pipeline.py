"""
RAG Pipeline: Retrieval-Augmented Generation with LangChain.

Builds a retrieval chain that searches the vector database for
relevant text/table/image documents and generates answers using
a local HuggingFace LLM.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline as hf_pipeline

from src.config import LLM_MODEL_NAME, LLM_MAX_NEW_TOKENS, TOP_K


# ── Prompt ───────────────────────────────────────────────────────────────────

RAG_PROMPT = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Use the following context from a document "
    "to answer the question. The context may include text, table data, and "
    "image descriptions.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _format_docs(docs):
    """Format retrieved documents into a single context string."""
    parts = []
    for doc in docs:
        doc_type = doc.metadata.get("type", "text")
        prefix = {"text": "📝", "table": "📊", "image": "🖼️"}.get(doc_type, "📝")
        parts.append(f"{prefix} [{doc_type.upper()}]:\n{doc.page_content}")
    return "\n\n".join(parts)


def _get_langchain_llm():
    """Wrap the HuggingFace pipeline in a LangChain-compatible LLM."""
    print(f"🧠 Loading LLM: {LLM_MODEL_NAME}...")
    pipe = hf_pipeline(
        "text2text-generation",
        model=LLM_MODEL_NAME,
        max_new_tokens=LLM_MAX_NEW_TOKENS,
    )
    return HuggingFacePipeline(pipeline=pipe)


# ── RAG Chain ────────────────────────────────────────────────────────────────

def build_rag_chain(vector_db):
    """
    Build a LangChain RAG chain:
        retriever → format docs → prompt → LLM → parse output

    The retriever searches across text chunks, table markdown,
    and image captions — making it truly multimodal.
    """
    retriever = vector_db.as_retriever(search_kwargs={"k": TOP_K})
    llm = _get_langchain_llm()

    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain


def query(chain, question: str):
    """Invoke the RAG chain and return the answer."""
    print(f"🔍 Querying: '{question}'")
    answer = chain.invoke(question)
    return answer


def search_documents(vector_db, question: str, k: int = TOP_K):
    """Search the vector DB directly and return matching documents."""
    retriever = vector_db.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(question)
    return docs
