"""
Retriever module for semantic search over the document corpus.
Bridges the embedding and vector store components.
"""
import logging
from typing import List, Dict, Any

import pandas as pd

from embedding import Embedder
from vector_store import VectorStore
from config import LOG_FORMAT, LOG_LEVEL, DEFAULT_TOP_K

# Setup module logger
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class Retriever:
    """
    Semantic retriever that finds relevant document chunks for a query.
    
    Combines the Embedder and VectorStore to perform end-to-end retrieval.
    """
    
    def __init__(
        self, 
        embedder: Embedder, 
        vector_store: VectorStore, 
        data: pd.DataFrame,
        text_column: str = "text_chunk"
    ) -> None:
        """
        Initialize the retriever.
        
        Args:
            embedder: Instance of Embedder for query encoding.
            vector_store: Instance of VectorStore with indexed documents.
            data: DataFrame containing the original text chunks.
            text_column: Name of the column containing text chunks.
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.data = data
        self.text_column = text_column
        
        logger.info(
            f"Retriever initialized with {len(data)} documents "
            f"and {vector_store.size} indexed vectors"
        )

    def search(self, query: str, k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
        """
        Search for the top k most similar documents to the query.
        
        Args:
            query: The search query string.
            k: Number of results to retrieve.
            
        Returns:
            List of dictionaries containing 'text', 'score', and 'index' keys.
        """
        logger.debug(f"Searching for: '{query}' (k={k})")
        
        # 1. Encode query
        query_vector = self.embedder.encode([query], show_progress=False)
        
        # 2. Search FAISS index
        scores, indices = self.vector_store.search(query_vector, k)
        
        # 3. Retrieve results from data
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:  # FAISS returns -1 if not enough results
                continue
            if idx >= len(self.data):
                logger.warning(f"Index {idx} out of bounds, skipping")
                continue
                
            results.append({
                "text": self.data.iloc[idx][self.text_column],
                "score": float(scores[0][i]),
                "index": int(idx)
            })
        
        logger.info(f"Retrieved {len(results)} results for query")
        return results

    def search_with_metadata(
        self, 
        query: str, 
        k: int = DEFAULT_TOP_K,
        metadata_columns: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search with additional metadata from the source DataFrame.
        
        Args:
            query: The search query string.
            k: Number of results to retrieve.
            metadata_columns: List of column names to include in results.
            
        Returns:
            List of dictionaries with text, score, index, and metadata.
        """
        results = self.search(query, k)
        
        if metadata_columns:
            for result in results:
                idx = result["index"]
                for col in metadata_columns:
                    if col in self.data.columns and col != self.text_column:
                        result[col] = self.data.iloc[idx][col]
                        
        return results


if __name__ == "__main__":
    # Test would go here, but we need the components initialized
    print("Retriever module loaded successfully")
