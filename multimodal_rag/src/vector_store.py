"""
Vector Store: Milvus Lite for local vector storage.

Uses langchain-milvus with a file-based Milvus Lite database,
replacing the remote vector store from the original tutorial.
"""

import os
from langchain_core.vectorstores import VectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import EMBEDDING_MODEL_NAME, MILVUS_DB_PATH, DATA_DIR


def get_embeddings():
    """Initialize the HuggingFace embedding model."""
    print(f"🔤 Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def create_milvus_store(embeddings=None):
    """
    Create a Milvus Lite vector store.
    Uses a local file-based database — no server needed.
    """
    from langchain_milvus import Milvus

    if embeddings is None:
        embeddings = get_embeddings()

    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"📦 Initializing Milvus Lite at: {MILVUS_DB_PATH}")

    vector_db = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": MILVUS_DB_PATH},
        auto_id=True,
        enable_dynamic_field=True,
        index_params={"index_type": "AUTOINDEX"},
    )
    return vector_db


def populate_store(vector_db, documents):
    """
    Add all documents (text chunks, tables, image descriptions)
    to the vector database.
    """
    if not documents:
        print("⚠️  No documents to add.")
        return []

    ids = vector_db.add_documents(documents)
    print(f"📦 Added {len(ids)} documents to the vector database.")
    return ids
