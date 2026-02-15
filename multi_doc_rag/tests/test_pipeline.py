import pytest
from src.pipeline import get_rag_chain, format_docs
from langchain_core.documents import Document

def test_format_docs():
    """Test document formatting."""
    docs = [
        Document(page_content="doc1"),
        Document(page_content="doc2")
    ]
    formatted = format_docs(docs)
    assert formatted == "doc1\n\ndoc2"

def test_get_rag_chain(mocker):
    """Test chain construction (mocked)."""
    mock_vector_store = mocker.Mock()
    mock_vector_store.as_retriever.return_value = mocker.Mock()
    
    # Mock ChatOllama to avoid API calls
    mocker.patch("src.pipeline.ChatOllama")
    
    chain = get_rag_chain(mock_vector_store)
    
    assert chain is not None
    mock_vector_store.as_retriever.assert_called_once()
