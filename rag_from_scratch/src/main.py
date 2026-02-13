import hashlib
import logging
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
setup_logging()
logger = logging.getLogger(__name__)


def _file_hash(path) -> str:
    """Return the MD5 hex digest of a file for staleness detection."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


HASH_SIDECAR = FAISS_INDEX_PATH.with_suffix(".hash")


def _index_is_stale(csv_path) -> bool:
    """Return True if the FAISS index does not match the current CSV."""
    if not HASH_SIDECAR.exists():
        return True
    stored = HASH_SIDECAR.read_text().strip()
    return stored != _file_hash(csv_path)


def initialize_system() -> RAGPipeline:
    """Initialize the RAG system components and orchestrator."""
    logger.info("Initializing system...")
    
    if not PROCESSED_SALARIES_PATH.exists():
        logger.info("Ingesting raw data...")
        df = run_ingestion()
    else:
        df = pd.read_csv(PROCESSED_SALARIES_PATH)
    
    embedder = Embedder()

    # Validate that config dimension matches the model
    assert embedder.dimension == EMBEDDING_DIMENSION, (
        f"Model dimension ({embedder.dimension}) != config EMBEDDING_DIMENSION ({EMBEDDING_DIMENSION})"
    )

    store = VectorStore(dimension=embedder.dimension)
    
    if not FAISS_INDEX_PATH.exists() or _index_is_stale(PROCESSED_SALARIES_PATH):
        if FAISS_INDEX_PATH.exists():
            logger.warning("FAISS index is stale — rebuilding.")
        logger.info("Building knowledge base index...")
        embeddings = embedder.encode(df['text_chunk'].tolist())
        store.add(embeddings)
        store.save(str(FAISS_INDEX_PATH))
        HASH_SIDECAR.write_text(_file_hash(PROCESSED_SALARIES_PATH))
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
