import os
from typing import Tuple
import faiss
import numpy as np


class VectorStore:

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)

    def add(self, vectors: np.ndarray) -> None:
        vectors = vectors.astype('float32', copy=False)
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension mismatch: expected {self.dimension}, got {vectors.shape[1]}")
        self.index.add(vectors)

    def search(self, query_vectors: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        query_vectors = query_vectors.astype('float32', copy=False)
        scores, indices = self.index.search(query_vectors, k)
        return scores, indices

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(self.index, path)

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Index not found at {path}")
        self.index = faiss.read_index(path)

    @property
    def size(self) -> int:
        return self.index.ntotal



