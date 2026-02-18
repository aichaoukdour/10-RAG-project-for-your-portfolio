"""
Configuration constants for the LangChain RAG project.
All models run locally via HuggingFace — no API keys required.
"""

# LLM Model (text-to-text generation)
LLM_MODEL_NAME = "google/flan-t5-base"
LLM_MAX_NEW_TOKENS = 256

# Embedding Model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Blog post to index
BLOG_URL = "https://lilianweng.github.io/posts/2023-06-23-agent/"

# Text splitter settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval settings
TOP_K = 3

# Agent routing keywords — queries containing these trigger document search
SEARCH_KEYWORDS = [
    "task", "decomposition", "agent", "planning", "memory",
    "tool", "llm", "autonomous", "cot", "chain of thought",
    "self-reflection", "react", "subgoal", "algorithm",
    "challenge", "limitation", "component", "prompt",
]
