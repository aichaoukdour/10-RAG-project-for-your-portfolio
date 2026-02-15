import os
import sys

# Ensure the app can find the src directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DOCS_DIRECTORY, CHROMA_PERSIST_DIRECTORY
from src.loader import load_documents
from src.chunker import split_text
from src.vector_store import create_vector_store, load_vector_store
from src.pipeline import query_rag_system

def main():
    print("=" * 50)
    print("  Multi-Document RAG System (Python + Ollama)")
    print("=" * 50)

    # Step 1: Handle Vector Database
    if not os.path.exists(CHROMA_PERSIST_DIRECTORY):
        print("📦 No vector DB found. Creating one...")
        docs = load_documents(DOCS_DIRECTORY)
        if not docs:
            print(f"❌ No PDF documents found in {DOCS_DIRECTORY}.")
            print("Please add some PDFs and restart the application.")
            return
            
        chunks = split_text(docs)
        vector_store = create_vector_store(chunks)
        print("✅ Vector database created successfully.")
    else:
        print("📦 Loading existing vector DB...")
        vector_store = load_vector_store()

    # Step 2: Interactive Chat Loop
    print("\n🚀 Ready! You can now ask questions about your documents.")
    print("Type 'exit' to quit.\n")
    
    while True:
        query = input("\n❓ Question: ").strip()
        if query.lower() == "exit":
            print("Goodbye!")
            break
        
        if not query:
            continue

        print("🤔 Thinking...")
        try:
            answer = query_rag_system(query, vector_store)
            print("\n🧠 Answer:\n", answer)
        except Exception as e:
            print(f"❌ Error during query: {e}")

if __name__ == "__main__":
    main()
