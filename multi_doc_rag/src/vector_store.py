from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from src.config import CHROMA_PERSIST_DIRECTORY, COLLECTION_NAME

# Initialize embeddings once
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def create_vector_store(chunks):
    """Creates a Chroma vector store from text chunks."""
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=CHROMA_PERSIST_DIRECTORY,
        collection_name=COLLECTION_NAME
    )
    print(f"📦 Vector database created and saved to {CHROMA_PERSIST_DIRECTORY}")
    return vector_store

def load_vector_store():
    """Loads an existing Chroma vector store."""
    return Chroma(
        persist_directory=CHROMA_PERSIST_DIRECTORY,
        embedding_function=embedding_function,
        collection_name=COLLECTION_NAME
    )
