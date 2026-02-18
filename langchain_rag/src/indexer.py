"""
Indexing pipeline: Load → Split → Store

Fetches the Lilian Weng blog post, splits it into chunks,
and stores embeddings in an InMemoryVectorStore.
"""

import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import (
    BLOG_URL,
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def load_blog_post(url: str = BLOG_URL):
    """
    Load the blog post HTML and extract relevant content
    using BeautifulSoup with a SoupStrainer.
    """
    bs4_strainer = bs4.SoupStrainer(
        class_=("post-title", "post-header", "post-content")
    )
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs={"parse_only": bs4_strainer},
    )
    docs = loader.load()
    print(f"📄 Loaded {len(docs)} document(s), "
          f"total characters: {len(docs[0].page_content)}")
    return docs


def split_documents(docs):
    """
    Split documents into smaller chunks for embedding and retrieval.
    Uses RecursiveCharacterTextSplitter as recommended for generic text.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    all_splits = text_splitter.split_documents(docs)
    print(f"✂️  Split into {len(all_splits)} sub-documents.")
    return all_splits


def create_vector_store(splits, embeddings):
    """
    Create an InMemoryVectorStore and add document splits.
    Returns the populated vector store.
    """
    vector_store = InMemoryVectorStore(embeddings)
    document_ids = vector_store.add_documents(documents=splits)
    print(f"📦 Stored {len(document_ids)} embeddings in vector store.")
    return vector_store


def get_embeddings():
    """Initialize the HuggingFace embedding model."""
    print(f"🔤 Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def build_index():
    """
    Full indexing pipeline: load → split → embed → store.
    Returns (vector_store, embeddings).
    """
    print("=" * 50)
    print("📚 Starting Indexing Pipeline")
    print("=" * 50)

    docs = load_blog_post()
    splits = split_documents(docs)
    embeddings = get_embeddings()
    vector_store = create_vector_store(splits, embeddings)

    print("✅ Indexing complete!\n")
    return vector_store, embeddings
