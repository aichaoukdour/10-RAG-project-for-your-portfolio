import os
import sys

# Ensure the app can find the src directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DOCS_DIRECTORY
from src.loader import load_and_split_docs
from src.pipeline import setup_retriever, rag_answer

def main():
    print("=" * 50)
    print("  Agentic RAG Pipeline (Flan-T5 + Chroma)")
    print("=" * 50)

    # Step 1: Data Ingestion
    chunks = load_and_split_docs(DOCS_DIRECTORY)
    if not chunks:
        print(f"⚠️ No PDF documents found in {DOCS_DIRECTORY}. System will only answer directly.")
    
    # Step 2: Setup Retriever
    retriever = setup_retriever(chunks)

    # Step 3: Interactive Chat Loop
    print("\n🚀 Ready! The Agent is watching your queries.")
    print("Type 'exit' to quit.\n")
    
    while True:
        query = input("\n❓ Question: ").strip()
        if query.lower() == "exit":
            print("Goodbye!")
            break
        
        if not query:
            continue

        print("🤔 Agent is thinking...")
        try:
            answer = rag_answer(query, retriever)
            print("\n🧠 Agent Response:\n", answer)
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
