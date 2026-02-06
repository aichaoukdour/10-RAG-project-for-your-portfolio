import os
import pandas as pd
from embedding import Embedder
from vector_store import VectorStore
from retriever import Retriever

def test_search():
    # 1. Setup paths
    processed_data_path = os.path.join('data', 'processed', 'cleaned_salaries.csv')
    index_path = os.path.join('data', 'processed', 'faiss_index.bin')

    if not os.path.exists(processed_data_path) or not os.path.exists(index_path):
        print("Error: Required data or index files not found. Please run src/main.py first.")
        return

    # 2. Load data and components
    print("Loading data and components...")
    df = pd.read_csv(processed_data_path)
    embedder = Embedder()
    
    # Initialize VectorStore and load index
    # We need to know the dimension; for all-MiniLM-L6-v2 it's 384
    store = VectorStore(dimension=384)
    store.load(index_path)

    retriever = Retriever(embedder, store, df)

    # 3. Perform tests
    queries = [
        "Data Scientist salary in the United States",
        "Entry level Machine Learning Engineer roles",
        "How much does a Senior Data Engineer earn in a medium sized company?",
        "Remote work for Data Analysts"
    ]

    print("\n--- Testing Semantic Search ---")
    for q in queries:
        print(f"\nQuery: {q}")
        results = retriever.search(q, k=3)
        for i, res in enumerate(results):
            print(f" {i+1}. {res}")

if __name__ == "__main__":
    test_search()
