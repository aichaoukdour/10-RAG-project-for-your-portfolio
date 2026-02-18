"""
Configuration constants for the Multimodal RAG project.
All models run locally via HuggingFace — no API keys required.
"""

import os

# ── LLM ──────────────────────────────────────────────────────────────────────
LLM_MODEL_NAME = "google/flan-t5-base"
LLM_MAX_NEW_TOKENS = 256

# ── Embeddings ───────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ── Image Captioning (Vision Model) ─────────────────────────────────────────
VISION_MODEL_NAME = "Salesforce/blip-image-captioning-base"

# ── Document Processing ─────────────────────────────────────────────────────
SAMPLE_PDF_URL = "https://midwestfoodbank.org/images/AR_2020_WEB2.pdf"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# ── Vector Store ─────────────────────────────────────────────────────────────
MILVUS_DB_PATH = os.path.join(DATA_DIR, "vectorstore.db")

# ── Retrieval ────────────────────────────────────────────────────────────────
TOP_K = 4
