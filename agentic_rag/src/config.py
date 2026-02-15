import os

# Model Configuration
LLM_MODEL_NAME = "google/flan-t5-base"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Paths
DOCS_DIRECTORY = os.path.join(os.getcwd(), "data")
CHROMA_PERSIST_DIRECTORY = os.path.join(os.getcwd(), "chroma_db")

# Routing Keywords
SEARCH_KEYWORDS = ["pdf", "document", "data", "summarize", "information", "find", "context", "file"]
