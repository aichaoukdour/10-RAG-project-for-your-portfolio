"""
Embedding module for converting text into vector representations.
Uses sentence-transformers for high-quality semantic embeddings.
"""
import logging
from typing import List, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME, LOG_FORMAT, LOG_LEVEL

# Setup module logger
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class Embedder:
    """
    Text embedding wrapper using sentence-transformers.
    
    Produces L2-normalized embeddings suitable for cosine similarity
    via inner product in FAISS.
    """
    
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME) -> None:
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the sentence-transformer model to load.
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def encode(self, texts: Union[str, List[str]], show_progress: bool = True) -> np.ndarray:
        """
        Convert text(s) into L2-normalized embedding vectors.
        
        Args:
            texts: Single string or list of strings to encode.
            show_progress: Whether to display encoding progress bar.
            
        Returns:
            np.ndarray: Array of shape (n_texts, embedding_dim) with normalized vectors.
        """
        if isinstance(texts, str):
            texts = [texts]
            
        logger.info(f"Encoding {len(texts)} text chunk(s)...")
        embeddings = self.model.encode(texts, show_progress_bar=show_progress)
        
        # Convert to numpy array and ensure float32 for FAISS
        embeddings = np.array(embeddings).astype('float32')
        
        # L2 Normalize for Cosine Similarity (equivalent to Inner Product)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        normalized = embeddings / norms
        
        logger.debug(f"Generated embeddings with shape: {normalized.shape}")
        return normalized

    @property
    def dimension(self) -> int:
        """Return the embedding dimension for this model."""
        return self.model.get_sentence_embedding_dimension()


if __name__ == "__main__":
    # Quick test
    embedder = Embedder()
    test_texts = ["This is a test sentence.", "Another test sentence."]
    vectors = embedder.encode(test_texts)
    print(f"Vectors shape: {vectors.shape}")
    print(f"Embedding dimension: {embedder.dimension}")
