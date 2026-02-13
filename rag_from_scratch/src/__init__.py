"""
RAG from Scratch - A complete Retrieval-Augmented Generation pipeline.

This package provides:
- Embedder: Text to vector embeddings using sentence-transformers
- VectorStore: FAISS-based vector storage and similarity search
- Retriever: Semantic search over stored documents
- Generator: LLM-based answer generation with fallback
- RAGPipeline: End-to-end orchestration of the RAG flow
"""
from .embedding import Embedder
from .vector_store import VectorStore
from .retriever import Retriever
from .generator import Generator, GenerationError, LocalAdvisor
from .pipeline import RAGPipeline
from .ingestion import load_data, clean_data
from .chunks import create_text_chunks

__all__ = [
    "Embedder",
    "VectorStore", 
    "Retriever",
    "Generator",
    "GenerationError",
    "LocalAdvisor",
    "RAGPipeline",
    "load_data",
    "clean_data",
    "create_text_chunks",
]

__version__ = "1.0.0"
