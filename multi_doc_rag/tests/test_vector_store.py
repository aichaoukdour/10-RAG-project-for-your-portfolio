import pytest
from src.vector_store import create_vector_store, load_vector_store
from langchain_core.documents import Document

def test_create_vector_store(mocker):
    """Test vector store creation with mocked Chroma."""
    mock_chroma = mocker.patch("src.vector_store.Chroma.from_documents")
    chunks = [Document(page_content="test chunk")]
    
    create_vector_store(chunks)
    
    mock_chroma.assert_called_once()
    args, kwargs = mock_chroma.call_args
    assert kwargs["documents"] == chunks
    assert kwargs["collection_name"] == "rag_docs"

def test_load_vector_store(mocker):
    """Test loading an existing vector store."""
    mock_chroma = mocker.patch("src.vector_store.Chroma")
    
    load_vector_store()
    
    mock_chroma.assert_called_once()
    args, kwargs = mock_chroma.call_args
    assert kwargs["collection_name"] == "rag_docs"
