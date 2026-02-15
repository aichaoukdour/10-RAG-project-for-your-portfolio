from transformers import pipeline
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import LLM_MODEL_NAME, EMBEDDING_MODEL_NAME, CHROMA_PERSIST_DIRECTORY
from src.controller import agent_controller

# Global instances (simplified for local execution)
_llm = None
_retriever = None

def get_llm():
    global _llm
    if _llm is None:
        print(f"🧠 Loading LLM: {LLM_MODEL_NAME}...")
        _llm = pipeline(
            "text2text-generation",
            model=LLM_MODEL_NAME,
            max_new_tokens=150
        )
    return _llm

def setup_retriever(chunks):
    """Sets up the vector store and retriever."""
    global _retriever
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    print("📦 Initializing Vector Store...")
    db = Chroma(
        collection_name="rag_store",
        embedding_function=embedding_model,
        persist_directory=CHROMA_PERSIST_DIRECTORY
    )
    
    if chunks:
        texts = [c.page_content for c in chunks]
        db.add_texts(texts)
    
    _retriever = db.as_retriever(search_kwargs={"k": 3})
    return _retriever

def rag_answer(query, retriever):
    """Execution loop with agentic routing."""
    action = agent_controller(query)
    llm = get_llm()

    if action == "search":
        print(f"🕵️ Agent decided to SEARCH document for: '{query}'")
        results = retriever.invoke(query)
        context = "\n".join([r.page_content for r in results])
        final_prompt = f"Use this context:\n{context}\n\nAnswer the question: {query}"
    else:
        print(f"🤖 Agent decided to answer DIRECTLY: '{query}'")
        final_prompt = query

    response = llm(final_prompt)[0]["generated_text"]
    return response
