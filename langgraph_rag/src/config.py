"""
Configuration for the LangGraph Retrieval Agent.
"""

# Models
LLM_MODEL = "llama3:8b"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Blog URLs to index
BLOG_URLS = [
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
]

# Splitting parameters
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Retrieval settings
RETRIEVAL_K = 3
