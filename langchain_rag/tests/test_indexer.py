"""
Tests for the indexing pipeline.
"""

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document


def test_split_documents():
    """Splitting a long document should produce multiple chunks."""
    from src.indexer import split_documents

    # Create a mock document with enough text to split
    long_text = "This is a test paragraph. " * 200  # ~5200 chars
    docs = [Document(page_content=long_text, metadata={"source": "test"})]

    splits = split_documents(docs)
    assert len(splits) > 1, "Expected multiple chunks from a long document"
    for split in splits:
        assert len(split.page_content) <= 1200  # chunk_size + some tolerance


def test_split_documents_preserves_metadata():
    """Each chunk should carry forward the original metadata."""
    from src.indexer import split_documents

    long_text = "Word " * 500
    docs = [Document(page_content=long_text, metadata={"source": "blog"})]

    splits = split_documents(docs)
    for split in splits:
        assert split.metadata["source"] == "blog"
        assert "start_index" in split.metadata


def test_create_vector_store():
    """Vector store should store documents and support similarity search."""
    from src.indexer import create_vector_store
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    splits = [
        Document(page_content="Task decomposition breaks tasks into steps.", metadata={}),
        Document(page_content="Python is a programming language.", metadata={}),
        Document(page_content="Chain of thought is a prompting technique.", metadata={}),
    ]

    vs = create_vector_store(splits, embeddings)

    # Search should return relevant results
    results = vs.similarity_search("task decomposition", k=2)
    assert len(results) == 2
    # The most relevant doc should mention task decomposition
    assert "task" in results[0].page_content.lower() or "decomposition" in results[0].page_content.lower()
