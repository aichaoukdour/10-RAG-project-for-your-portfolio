"""
Tests for the embedding module.
"""
import pytest
import numpy as np

from embedding import Embedder


class TestEmbedder:
    """Tests for the Embedder class."""
    
    @pytest.fixture(scope="class")
    def embedder(self):
        """Create a shared embedder instance (model loading is slow)."""
        return Embedder()
    
    def test_initialization(self, embedder):
        """Test that embedder initializes correctly."""
        assert embedder.model is not None
        assert embedder.dimension > 0
    
    def test_dimension_property(self, embedder):
        """Test that dimension property returns expected value."""
        # all-MiniLM-L6-v2 has 384 dimensions
        assert embedder.dimension == 384
    
    def test_encode_single_text(self, embedder):
        """Test encoding a single text string."""
        text = "This is a test sentence."
        vectors = embedder.encode(text)
        
        assert isinstance(vectors, np.ndarray)
        assert vectors.dtype == np.float32
        assert vectors.shape == (1, 384)
    
    def test_encode_multiple_texts(self, embedder):
        """Test encoding multiple texts."""
        texts = ["First sentence.", "Second sentence.", "Third sentence."]
        vectors = embedder.encode(texts)
        
        assert vectors.shape == (3, 384)
    
    def test_vectors_are_normalized(self, embedder):
        """Test that output vectors are L2 normalized."""
        texts = ["Test sentence one.", "Test sentence two."]
        vectors = embedder.encode(texts)
        
        # L2 norm of each vector should be approximately 1
        norms = np.linalg.norm(vectors, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-5)
    
    def test_similar_texts_have_high_similarity(self, embedder):
        """Test that similar texts produce similar embeddings."""
        texts = [
            "Data Scientist salary in New York",
            "How much does a Data Scientist earn in NY?"
        ]
        vectors = embedder.encode(texts)
        
        # Cosine similarity (dot product of normalized vectors)
        similarity = np.dot(vectors[0], vectors[1])
        assert similarity > 0.7  # Similar texts should have high similarity
    
    def test_different_texts_have_lower_similarity(self, embedder):
        """Test that unrelated texts have lower similarity."""
        texts = [
            "Data Scientist salary in New York",
            "The weather is sunny today"
        ]
        vectors = embedder.encode(texts)
        
        similarity = np.dot(vectors[0], vectors[1])
        assert similarity < 0.5  # Unrelated texts should have lower similarity
