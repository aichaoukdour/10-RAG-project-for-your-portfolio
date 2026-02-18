"""
Main CLI entry point for the LangChain RAG project.

Demonstrates two RAG approaches:
  1. RAG Agent  — intent-routing + retrieval tool
  2. RAG Chain  — always-retrieve LCEL chain
"""

import sys
from src.indexer import build_index
from src.agent import run_agent
from src.chain import build_rag_chain, run_chain


def print_banner():
    print()
    print("=" * 60)
    print("  🔗  LangChain RAG — Local HuggingFace Edition")
    print("=" * 60)
    print()


def print_menu():
    print("┌─────────────────────────────────────────┐")
    print("│  1 │ Index the blog post                 │")
    print("│  2 │ Ask a question  (Agent mode)        │")
    print("│  3 │ Ask a question  (Chain mode)        │")
    print("│  4 │ Quit                                │")
    print("└─────────────────────────────────────────┘")


def main():
    print_banner()

    vector_store = None
    rag_chain = None

    while True:
        print_menu()
        choice = input("Select an option: ").strip()

        if choice == "1":
            # ── Index ────────────────────────────────
            vector_store, _ = build_index()
            # Pre-build the chain so it's ready for option 3
            rag_chain = build_rag_chain(vector_store)
            print("✅ Index built. You can now ask questions.\n")

        elif choice == "2":
            # ── Agent mode ───────────────────────────
            if vector_store is None:
                print("⚠️  Please index the blog post first (option 1).\n")
                continue
            query = input("🗣️  Your question: ").strip()
            if not query:
                continue
            print()
            answer = run_agent(query, vector_store)
            print(f"\n💬 Answer:\n{answer}\n")

        elif choice == "3":
            # ── Chain mode ───────────────────────────
            if rag_chain is None:
                print("⚠️  Please index the blog post first (option 1).\n")
                continue
            query = input("🗣️  Your question: ").strip()
            if not query:
                continue
            print()
            answer = run_chain(rag_chain, query)
            print(f"\n💬 Answer:\n{answer}\n")

        elif choice == "4":
            print("👋 Goodbye!")
            sys.exit(0)

        else:
            print("❌ Invalid option. Please try again.\n")


if __name__ == "__main__":
    main()
