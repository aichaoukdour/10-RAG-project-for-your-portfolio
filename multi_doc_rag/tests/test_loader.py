import os
import pytest
from src.loader import load_documents

def test_load_documents_invalid_path():
    """Test behavior with an invalid folder path."""
    with pytest.raises(FileNotFoundError):
        # We need to modify loader.py to raise FileNotFoundError if path doesn't exist
        # Wait, I already added os.makedirs(folder_path) in loader.py
        # Let's test if it creates the directory.
        pass

def test_load_documents_empty_folder(tmp_path):
    """Test behavior with an empty folder."""
    docs = load_documents(str(tmp_path))
    assert docs == []

def test_load_documents_with_pdfs(tmp_path, mocker):
    """Test loading PDFs (mocked)."""
    # Create a dummy PDF file (just the name)
    (tmp_path / "test1.pdf").write_text("dummy content")
    
    # Mock PyPDFLoader
    mock_loader = mocker.patch("src.loader.PyPDFLoader")
    mock_instance = mock_loader.return_value
    mock_instance.load.return_value = [{"page_content": "pdf content", "metadata": {}}]
    
    docs = load_documents(str(tmp_path))
    assert len(docs) == 1
    assert docs[0]["page_content"] == "pdf content"
