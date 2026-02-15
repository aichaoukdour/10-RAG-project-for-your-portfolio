import os

# Ollama configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")

# Local storage
CHROMA_PERSIST_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "rag_docs"

# Data paths
DOCS_DIRECTORY = "./data"
