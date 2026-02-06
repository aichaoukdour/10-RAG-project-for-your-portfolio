from sentence_transformers import SentenceTransformer
import numpy as np

class Embedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """Initialize the embedding model."""
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        """Convert a list of strings into a numpy array of embeddings."""
        print(f"Encoding {len(texts)} text chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return np.array(embeddings).astype('float32')

if __name__ == "__main__":
    # Quick test
    embedder = Embedder()
    test_texts = ["This is a test sentence.", "Another test sentence."]
    vectors = embedder.encode(test_texts)
    print(f"Vectors shape: {vectors.shape}")
