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
    EMBEDDING_DIMENSION, setup_logging
)

load_dotenv()
logger = setup_logging()

def initialize_system() -> RAGPipeline:
    """Initialize the RAG system components and orchestrator."""
    logger.info("Initializing system...")
    
    if not PROCESSED_SALARIES_PATH.exists():
        logger.info("Ingesting raw data...")
        df = run_ingestion()
    else:
        df = pd.read_csv(PROCESSED_SALARIES_PATH)
    
    embedder = Embedder()
    store = VectorStore(dimension=EMBEDDING_DIMENSION)
    
    if not FAISS_INDEX_PATH.exists():
        logger.info("Building knowledge base index...")
        embeddings = embedder.encode(df['text_chunk'].tolist())
        store.add(embeddings)
        store.save(str(FAISS_INDEX_PATH))
    else:
        store.load(str(FAISS_INDEX_PATH))
    
    return RAGPipeline(embedder, store, df)

def handle_query(advisor: RAGPipeline) -> None:
    """Handle a user question about careers/salaries."""
    query = input("\nQuestion: ").strip()
    if not query:
        return
        
    print("Searching...")
    try:
        result = advisor.run(query)
        print(f"\nResponse:\n{result['answer']}")
        print(f"\n[Source: {result['source']} | Chunks used: {len(result['context'])}]")
    except Exception as e:
        logger.error(f"Query error: {e}")
        print(f"Error: {e}")

def handle_insight_report(advisor: RAGPipeline) -> None:
    """Handle a salary insight report request."""
    job_title = input("\nJob Title: ").strip()
    if not job_title:
        return
        
    print(f"Analyzing {job_title} data...")
    try:
        result = advisor.get_salary_insight(job_title)
        print(f"\nSalary Insight Report: {result['job_title']}")
        print("-" * 30)
        print(result['report'])
        print(f"\n[Based on {result['num_records_analyzed']} records]")
    except Exception as e:
        logger.error(f"Insight error: {e}")
        print(f"Error: {e}")

def run_interactive_session(advisor: RAGPipeline) -> None:
    """Main interactive loop."""
    menu = """
Main Menu:
1. Ask a question
2. Salary Insight Report
3. Exit
"""
    while True:
        print(menu)
        try:
            choice = input("Select (1-3): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if choice == '1':
            handle_query(advisor)
        elif choice == '2':
            handle_insight_report(advisor)
        elif choice in ('3', 'exit', 'quit', 'q'):
            print("Goodbye.")
            break
        else:
            print("Invalid selection.")

def main() -> int:
    """Main entry point."""
    print("\nAI Career Advisor (RAG)")
    print("-" * 23)
    
    try:
        advisor = initialize_system()
        run_interactive_session(advisor)
        return 0
    except FileNotFoundError as e:
        print(f"Error: Data file not found at {RAW_SALARIES_PATH}")
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 0
    except Exception as e:
        logger.exception("Application crash")
        print(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
