"""
Main entry point for the RAG Career Advisor application.
Provides an interactive CLI for querying salary and career information.
"""
import logging
import os
import sys

import pandas as pd
from dotenv import load_dotenv

from ingestion import load_data, clean_data, create_text_chunks, save_processed_data
from embedding import Embedder
from vector_store import VectorStore
from pipeline import RAGPipeline
from config import (
    LOG_FORMAT, LOG_LEVEL,
    PROCESSED_SALARIES_PATH, FAISS_INDEX_PATH, RAW_SALARIES_PATH,
    EMBEDDING_DIMENSION, OPENAI_API_KEY
)

# Load environment variables from .env file
load_dotenv()

# Setup module logger
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def check_api_key() -> bool:
    """Check and warn about API key status."""
    if not OPENAI_API_KEY:
        print("\n" + "=" * 60)
        print("⚠️  WARNING: OPENAI_API_KEY not found in environment")
        print("=" * 60)
        print("\nTo use the full LLM features, create a '.env' file with:")
        print("  OPENAI_API_KEY=sk-your-key-here")
        print("\nThe system will use the LocalAdvisor fallback for now.")
        print("=" * 60 + "\n")
        return False
    return True


def initialize_system() -> RAGPipeline:
    """
    Initialize or load all RAG components.
    
    Returns:
        RAGPipeline: Fully initialized pipeline ready for queries.
        
    Raises:
        FileNotFoundError: If raw data doesn't exist and no processed data.
    """
    logger.info("Initializing RAG system...")
    
    # Check API key status
    check_api_key()
    
    # Convert Path objects to strings for compatibility
    processed_path = str(PROCESSED_SALARIES_PATH)
    index_path = str(FAISS_INDEX_PATH)
    raw_path = str(RAW_SALARIES_PATH)
    
    # 1. Data Ingestion & Cleaning
    if not os.path.exists(processed_path):
        logger.info("Processed data not found. Running data ingestion...")
        df = load_data(raw_path)
        df = clean_data(df)
        df = create_text_chunks(df)
        save_processed_data(df, processed_path)
    else:
        logger.info(f"Loading processed data from {processed_path}")
        df = pd.read_csv(processed_path)
        logger.info(f"Loaded {len(df)} records")
    
    # 2. Initialize Embedder
    embedder = Embedder()
    
    # 3. Build or Load FAISS Index
    if not os.path.exists(index_path):
        logger.info("Building FAISS index (this may take a moment)...")
        text_chunks = df['text_chunk'].tolist()
        embeddings = embedder.encode(text_chunks)
        
        dimension = embeddings.shape[1]
        store = VectorStore(dimension)
        store.add(embeddings)
        store.save(index_path)
    else:
        logger.info(f"Loading existing FAISS index from {index_path}")
        store = VectorStore(dimension=EMBEDDING_DIMENSION)
        store.load(index_path)
    
    # 4. Create and return pipeline
    pipeline = RAGPipeline(embedder, store, df)
    logger.info("RAG system initialized successfully!")
    return pipeline


def display_menu() -> None:
    """Display the main menu options."""
    print("\n" + "-" * 40)
    print("Options:")
    print("  1. Ask a question about careers/salaries")
    print("  2. Get a Salary Insight Report")
    print("  3. Exit")
    print("-" * 40)


def run_interactive_session(advisor: RAGPipeline) -> None:
    """
    Run the interactive question-answering loop.
    
    Args:
        advisor: Initialized RAGPipeline instance.
    """
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
