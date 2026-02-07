"""
Vector store module using FAISS for efficient similarity search.
Supports saving/loading indices and provides a clean interface.
"""
import logging
import os
from typing import Tuple

import faiss
import numpy as np



# Setup module logger
# Setup module logger
logger = logging.getLogger(__name__)


class VectorStore:
    """
    FAISS-based vector store for similarity search.
    
    Uses Inner Product (IndexFlatIP) which is equivalent to cosine similarity
    when vectors are L2-normalized.
    """
    
    def __init__(self, dimension: int) -> None:
        """
        Initialize the FAISS index.
        
        Args:
            dimension: Dimensionality of the vectors to store.
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        logger.info(f"Initialized FAISS IndexFlatIP with dimension={dimension}")

    def add(self, vectors: np.ndarray) -> None:
        """
        Add vectors to the index.
        
        Args:
            vectors: Numpy array of shape (n_vectors, dimension) with float32 dtype.
                     Vectors should be L2-normalized for cosine similarity.
        
        Raises:
            ValueError: If vectors have wrong dimension or dtype.
        """
        if vectors.dtype != np.float32:
            logger.warning(f"Converting vectors from {vectors.dtype} to float32")
            vectors = vectors.astype('float32')
            
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.dimension}, got {vectors.shape[1]}"
            )
            
        logger.info(f"Adding {len(vectors)} vectors to FAISS index...")
        self.index.add(vectors)
        logger.info(f"Index now contains {self.index.ntotal} vectors")

    def search(self, query_vectors: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for the k most similar vectors.
        
        Args:
            query_vectors: Query vectors of shape (n_queries, dimension).
            k: Number of nearest neighbors to retrieve.
            
        Returns:
            Tuple of (scores, indices) arrays, each of shape (n_queries, k).
            Higher scores indicate greater similarity.
        """
        if query_vectors.dtype != np.float32:
            query_vectors = query_vectors.astype('float32')
            
        scores, indices = self.index.search(query_vectors, k)
        return scores, indices

    def save(self, path: str) -> None:
        """
        Save the FAISS index to disk.
        
        Args:
            path: File path to save the index.
        """
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(self.index, path)
        logger.info(f"FAISS index saved to {path} ({self.index.ntotal} vectors)")

    def load(self, path: str) -> None:
        """
        Load a FAISS index from disk.
        
        Args:
            path: File path to load the index from.
            
        Raises:
            FileNotFoundError: If the index file doesn't exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Index not found at {path}")
        self.index = faiss.read_index(path)
        logger.info(f"FAISS index loaded from {path} ({self.index.ntotal} vectors)")

    @property
    def size(self) -> int:
        """Return the number of vectors in the index."""
        return self.index.ntotal


if __name__ == "__main__":
    # Quick test
    from config import EMBEDDING_DIMENSION
    
    dim = EMBEDDING_DIMENSION
    store = VectorStore(dim)
    
    dummy_vectors = np.random.random((5, dim)).astype('float32')
    # Normalize for cosine similarity
    norms = np.linalg.norm(dummy_vectors, axis=1, keepdims=True)
    dummy_vectors = dummy_vectors / norms
    
    store.add(dummy_vectors)
    print(f"Index size: {store.size}")
    
    # Test search
    query = dummy_vectors[:1]
    scores, indices = store.search(query, k=3)
    print(f"Top 3 matches: indices={indices[0]}, scores={scores[0]}")
    
    # Test save/load
    store.save("test_index.bin")
    store.load("test_index.bin")
    os.remove("test_index.bin")
    print("Save/load test passed!")
