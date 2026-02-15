from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.config import OLLAMA_BASE_URL, LLM_MODEL

def format_docs(docs):
    """Formats retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain(vector_store):
    """Constructs the RAG chain."""
    llm = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=LLM_MODEL
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful assistant.
        Answer ONLY using the context below.
        If the answer is not present in the context, say "I don't know."

        Context:
        {context}

        Question:
        {question}
        """
    )

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain

def query_rag_system(query_text, vector_store):
    """Invokes the RAG chain for a user query."""
    chain = get_rag_chain(vector_store)
    return chain.invoke(query_text)
