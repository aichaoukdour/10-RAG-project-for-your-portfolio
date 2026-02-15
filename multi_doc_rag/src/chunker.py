from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List

def split_text(documents: List):
    """Splits documents into smaller chunks for processing."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(documents)
    print(f"✂️ Created {len(chunks)} chunks.")
    return chunks
