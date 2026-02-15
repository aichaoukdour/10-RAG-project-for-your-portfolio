import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import DOCS_DIRECTORY

def load_and_split_docs(folder_path=DOCS_DIRECTORY):
    """Loads PDFs from a folder and splits them into chunks."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"📁 Created directory: {folder_path}")
        return []

    docs = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            file_path = os.path.join(folder_path, file)
            print(f"📄 Loading: {file}")
            try:
                loader = PyPDFLoader(file_path)
                docs.extend(loader.load())
            except Exception as e:
                print(f"❌ Error loading {file}: {e}")
    
    if not docs:
        return []

    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80
    )
    chunks = text_splitter.split_documents(docs)
    print(f"✂️ Created {len(chunks)} chunks.")
    return chunks
