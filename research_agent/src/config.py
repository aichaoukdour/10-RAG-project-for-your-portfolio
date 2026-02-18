"""
Configuration constants for the AI Research Agent.
"""

# Search settings
SEARCH_RESULTS = 6        # How many URLs to check
PASSAGES_PER_PAGE = 4     # How many passages to pull from each URL
TIMEOUT = 8               # How long to wait for a webpage to load

# Ranking / Summary settings
TOP_K_PASSAGES = 5        # How many relevant passages to use for the summary
SUMMARY_SENTENCES = 3     # How many sentences in the final summary

# AI Models
# Using a fast, high-quality local embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
