import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from src.config import DOCS_DIRECTORY

def load_documents(folder_path: str = DOCS_DIRECTORY):
    """Loads all PDF documents from the specified folder."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"📁 Created directory: {folder_path}")
        return []

    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            print(f"📄 Loading: {filename}")
            try:
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            except Exception as e:
                print(f"❌ Error loading {filename}: {e}")
    
    print(f"✅ Loaded {len(documents)} document pages.")
    return documents
