from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from .config import BLOG_URLS, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL_NAME

def get_retriever():
    """
    Fetches documents, splits them, and returns a retriever.
    """
    print("📚 Loading and indexing documents...")
    
    # 1. Load documents
    docs = []
    for url in BLOG_URLS:
        try:
            loader = WebBaseLoader(url)
            docs.extend(loader.load())
        except Exception as e:
            print(f"⚠️ Failed to load {url}: {e}")
            
    if not docs:
        raise ValueError("No documents were loaded. Check your internet connection or URLs.")

    # 2. Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP
    )
    doc_splits = text_splitter.split_documents(docs)
    print(f"✂️ Split into {len(doc_splits)} chunks.")

    # 3. Create vector store and retriever
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = InMemoryVectorStore.from_documents(
        documents=doc_splits, 
        embedding=embeddings
    )
    
    return vectorstore.as_retriever()
