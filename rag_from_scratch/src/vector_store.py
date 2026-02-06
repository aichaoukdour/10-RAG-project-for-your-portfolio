import faiss
import os
import numpy as np

class VectorStore:
    def __init__(self, dimension):
        """Initialize the FAISS index."""
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)

    def add(self, vectors):
        """Add vectors to the index."""
        print(f"Adding {len(vectors)} vectors to FAISS index...")
        self.index.add(vectors)

    def save(self, path):
        """Save the FAISS index to disk."""
        faiss.write_index(self.index, path)
        print(f"FAISS index saved to {path}")

    def load(self, path):
        """Load the FAISS index from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Index not found at {path}")
        self.index = faiss.read_index(path)
        print(f"FAISS index loaded from {path}")

if __name__ == "__main__":
    # Quick test
    dim = 384
    store = VectorStore(dim)
    dummy_vectors = np.random.random((5, dim)).astype('float32')
    store.add(dummy_vectors)
    store.save("test_index.bin")
    os.remove("test_index.bin")
