import os
import pandas as pd
from ingestion import load_data, clean_data, create_text_chunks, save_processed_data
from embedding import Embedder
from vector_store import VectorStore

def main():
    # Paths
    raw_data_path = os.path.join('data', 'raw', 'salaries.csv')
    processed_data_path = os.path.join('data', 'processed', 'cleaned_salaries.csv')
    index_path = os.path.join('data', 'processed', 'faiss_index.bin')

    # 1. Data Ingestion & Cleaning
    if not os.path.exists(processed_data_path):
        print("--- Start Data Ingestion ---")
        df = load_data(raw_data_path)
        df = clean_data(df)
        df = create_text_chunks(df)
        save_processed_data(df, processed_data_path)
    else:
        print(f"--- Loading Processed Data from {processed_data_path} ---")
        df = pd.read_csv(processed_data_path)

    # 2. Embedding Generation
    print("--- Start Embedding Generation ---")
    embedder = Embedder()
    # We only embed the unique text chunks
    text_chunks = df['text_chunk'].tolist()
    embeddings = embedder.encode(text_chunks)

    # 3. Vector Storage
    print("--- Start Vector Storage ---")
    dimension = embeddings.shape[1]
    store = VectorStore(dimension)
    store.add(embeddings)
    store.save(index_path)

    print("\n--- RAG Pipeline Initialized Successfully ---")
    print(f"Total records indexed: {len(df)}")
    print(f"Index Location: {index_path}")

if __name__ == "__main__":
    main()
