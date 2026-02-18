"""
Tests for the RAG pipeline.
"""

import pytest
from langchain_core.documents import Document


class TestFormatDocs:
    """Tests for document formatting."""

    def test_format_docs_text(self):
        """Text documents should be formatted with text prefix."""
        from src.rag_pipeline import _format_docs
        docs = [Document(page_content="Hello world", metadata={"type": "text"})]
        result = _format_docs(docs)
        assert "[TEXT]" in result
        assert "Hello world" in result

    def test_format_docs_table(self):
        """Table documents should be formatted with table prefix."""
        from src.rag_pipeline import _format_docs
        docs = [Document(page_content="| A | B |", metadata={"type": "table"})]
        result = _format_docs(docs)
        assert "[TABLE]" in result

    def test_format_docs_image(self):
        """Image documents should be formatted with image prefix."""
        from src.rag_pipeline import _format_docs
        docs = [Document(page_content="A red car", metadata={"type": "image"})]
        result = _format_docs(docs)
        assert "[IMAGE]" in result

    def test_format_docs_mixed(self):
        """Mixed document types should all appear in output."""
        from src.rag_pipeline import _format_docs
        docs = [
            Document(page_content="Text content", metadata={"type": "text"}),
            Document(page_content="| col |", metadata={"type": "table"}),
            Document(page_content="A photo", metadata={"type": "image"}),
        ]
        result = _format_docs(docs)
        assert "[TEXT]" in result
        assert "[TABLE]" in result
        assert "[IMAGE]" in result
