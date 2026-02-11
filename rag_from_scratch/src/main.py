import logging
import os
import sys

import pandas as pd
from dotenv import load_dotenv

from ingestion import run_ingestion
from embedding import Embedder
from vector_store import VectorStore
from pipeline import RAGPipeline
from config import (
    LOG_FORMAT, LOG_LEVEL,
    PROCESSED_SALARIES_PATH, FAISS_INDEX_PATH, RAW_SALARIES_PATH,
    EMBEDDING_DIMENSION, OPENAI_API_KEY, setup_logging
)


load_dotenv()

logger = setup_logging()


def initialize_system() -> RAGPipeline:
    """Initialize the RAG system components and orchestrator."""
    logger.info("Initializing RAG system...")
    
    # 1. Data Ingestion & Cleaning
    if not PROCESSED_SALARIES_PATH.exists():
        logger.info("Processed data not found. Running data ingestion...")
        df = run_ingestion()
    else:
        logger.info(f"Loading processed data from {PROCESSED_SALARIES_PATH}")
        df = pd.read_csv(PROCESSED_SALARIES_PATH)
        logger.info(f"Loaded {len(df)} records")
    
    # 2. Initialize Embedder
    embedder = Embedder()
    
    # 3. Build or Load FAISS Index
    store = VectorStore(dimension=EMBEDDING_DIMENSION)
    if not FAISS_INDEX_PATH.exists():
        logger.info("Building FAISS index (this may take a moment)...")
        embeddings = embedder.encode(df['text_chunk'].tolist())
        store.add(embeddings)
        store.save(str(FAISS_INDEX_PATH))
    else:
        logger.info(f"Loading existing FAISS index from {FAISS_INDEX_PATH}")
        store.load(str(FAISS_INDEX_PATH))
    
    # 4. Create pipeline
    return RAGPipeline(embedder, store, df)


def display_menu() -> None:
    """Display the main menu options."""
    print("\n" + "-" * 40)
    print("Options:")
    print("  1. Ask a question about careers/salaries")
    print("  2. Get a Salary Insight Report")
    print("  3. Exit")
    print("-" * 40)


def run_interactive_session(advisor: RAGPipeline) -> None:
    while True:
        display_menu()
        
        try:
            choice = input("\nSelect an option (1-3): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break
        
        if choice == '1':
            try:
                query = input("\nWhat would you like to know? ").strip()
                if not query:
                    print("Please enter a question.")
                    continue
                    
                print("\nSearching knowledge base...")
                result = advisor.run(query)
                
                print(f"\n{'=' * 50}")
                print("ADVISOR:")
                print(f"{'=' * 50}")
                print(result['answer'])
                print(f"\n[Source: {result['source']} | Chunks: {len(result['context'])}]")
                
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                print(f"\nSorry, an error occurred: {e}")
                
        elif choice == '2':
            try:
                job_title = input("\nEnter a job title (e.g., 'Data Scientist'): ").strip()
                if not job_title:
                    print("Please enter a job title.")
                    continue
                    
                print(f"\nAnalyzing salary data for: {job_title}...")
                result = advisor.get_salary_insight(job_title)
                
                print(f"\n{'=' * 50}")
                print(f"SALARY INSIGHT REPORT: {result['job_title']}")
                print(f"{'=' * 50}")
                print(result['report'])
                print(f"\n[Based on {result['num_records_analyzed']} relevant records]")
                
            except Exception as e:
                logger.error(f"Error generating insight report: {e}")
                print(f"\nSorry, an error occurred: {e}")
                
        elif choice in ('3', 'exit', 'quit', 'q'):
            print("\nGoodbye! 👋")
            break
            
        else:
            print("\n❌ Invalid choice. Please enter 1, 2, or 3.")


def main() -> int:
    """
    Main entry point for the application.
    
    Returns:
        int: Exit code (0 for success, 1 for error).
    """
    print("\n" + "=" * 60)
    print("🚀 Welcome to the AI Career Advisor (RAG Edition)")
    print("=" * 60)
    
    try:
        advisor = initialize_system()
        run_interactive_session(advisor)
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        print(f"\n❌ Error: {e}")
        print("Please ensure the raw data file exists at:")
        print(f"  {RAW_SALARIES_PATH}")
        return 1
        
    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye!")
        return 0
        
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        print(f"\n❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
