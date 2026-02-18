"""
Main CLI entry point for the Multimodal RAG project.

Demonstrates a complete multimodal RAG pipeline:
  1. Parse PDF → text chunks + tables + images
  2. Caption images with BLIP
  3. Store all in Milvus vector DB
  4. Query with LangChain RAG chain
"""

import os
import sys
import itertools

from src.config import DATA_DIR
from src.doc_processor import download_sample_pdf, process_pdf
from src.image_captioner import caption_images
from src.vector_store import get_embeddings, create_milvus_store, populate_store
from src.rag_pipeline import build_rag_chain, query, search_documents


def print_banner():
    print()
    print("=" * 60)
    print("  🖼️  Multimodal RAG — Local HuggingFace Edition")
    print("=" * 60)
    print()


def print_menu():
    print("┌──────────────────────────────────────────────┐")
    print("│  1 │ Download & process sample PDF            │")
    print("│  2 │ Process a local PDF from data/           │")
    print("│  3 │ Ask a question                           │")
    print("│  4 │ Search documents (no generation)         │")
    print("│  5 │ Quit                                     │")
    print("└──────────────────────────────────────────────┘")


def process_and_index(source, vector_db, embeddings):
    """Full pipeline: process PDF → caption images → index in vector store."""
    print("\n" + "=" * 50)
    print("📚 Processing Document")
    print("=" * 50)

    # Step 1: Process PDF with Docling
    text_docs, table_docs, images = process_pdf(source)

    # Step 2: Caption images with BLIP
    start_id = len(text_docs) + len(table_docs)
    image_docs = caption_images(images, start_id=start_id)

    # Step 3: Combine all documents
    all_docs = list(itertools.chain(text_docs, table_docs, image_docs))
    print(f"\n📋 Summary: {len(text_docs)} text chunks, "
          f"{len(table_docs)} tables, {len(image_docs)} image descriptions")

    # Step 4: Store in vector DB
    populate_store(vector_db, all_docs)

    print("✅ Document indexed successfully!\n")
    return all_docs


def main():
    print_banner()

    embeddings = get_embeddings()
    vector_db = create_milvus_store(embeddings)
    rag_chain = None
    indexed = False

    while True:
        print_menu()
        choice = input("Select an option: ").strip()

        if choice == "1":
            # ── Download & process sample PDF ────────────
            pdf_path = download_sample_pdf()
            process_and_index(pdf_path, vector_db, embeddings)
            rag_chain = build_rag_chain(vector_db)
            indexed = True

        elif choice == "2":
            # ── Process a local PDF ──────────────────────
            os.makedirs(DATA_DIR, exist_ok=True)
            pdfs = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
            if not pdfs:
                print(f"⚠️  No PDFs found in {DATA_DIR}. Add PDFs and try again.\n")
                continue

            print("\nAvailable PDFs:")
            for i, pdf_name in enumerate(pdfs, 1):
                print(f"  {i}. {pdf_name}")
            idx = input("Select a PDF number: ").strip()
            try:
                pdf_path = os.path.join(DATA_DIR, pdfs[int(idx) - 1])
            except (ValueError, IndexError):
                print("❌ Invalid selection.\n")
                continue

            process_and_index(pdf_path, vector_db, embeddings)
            rag_chain = build_rag_chain(vector_db)
            indexed = True

        elif choice == "3":
            # ── Ask a question ───────────────────────────
            if not indexed:
                print("⚠️  Please process a PDF first (option 1 or 2).\n")
                continue
            question = input("🗣️  Your question: ").strip()
            if not question:
                continue
            print()
            answer = query(rag_chain, question)
            print(f"\n💬 Answer:\n{answer}\n")

        elif choice == "4":
            # ── Search documents ─────────────────────────
            if not indexed:
                print("⚠️  Please process a PDF first (option 1 or 2).\n")
                continue
            question = input("🔍 Search query: ").strip()
            if not question:
                continue
            print()
            docs = search_documents(vector_db, question)
            for i, doc in enumerate(docs, 1):
                doc_type = doc.metadata.get("type", "text")
                print(f"── Result {i} [{doc_type.upper()}] ──")
                print(doc.page_content[:300])
                print()

        elif choice == "5":
            print("👋 Goodbye!")
            sys.exit(0)

        else:
            print("❌ Invalid option. Please try again.\n")


if __name__ == "__main__":
    main()
