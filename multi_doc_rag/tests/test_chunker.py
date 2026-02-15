from src.chunker import split_text
from langchain_core.documents import Document

def test_split_text():
    """Test text splitting logic."""
    docs = [
        Document(page_content="This is a long sentence that should be split into chunks if the size is small enough. " * 10)
    ]
    chunks = split_text(docs)
    assert len(chunks) > 0
    assert isinstance(chunks[0], Document)
    assert len(chunks[0].page_content) <= 1000
