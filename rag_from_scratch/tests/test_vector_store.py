"""
Tests for the vector_store module.
"""
import os
import tempfile
import pytest
import numpy as np

from vector_store import VectorStore


class TestVectorStore:
    """Tests for the VectorStore class."""
    
    def test_initialization(self):
        """Test that VectorStore initializes with correct dimension."""
        store = VectorStore(dimension=384)
        
        assert store.dimension == 384
        assert store.size == 0
    
    def test_add_vectors(self, sample_embeddings):
        """Test adding vectors to the store."""
        store = VectorStore(dimension=384)
        store.add(sample_embeddings)
        
        assert store.size == 5
    
    def test_add_wrong_dimension_raises(self):
        """Test that adding vectors with wrong dimension raises ValueError."""
        store = VectorStore(dimension=384)
        wrong_dim_vectors = np.random.random((3, 256)).astype('float32')
        
        with pytest.raises(ValueError, match="dimension mismatch"):
            store.add(wrong_dim_vectors)
    
    def test_search_returns_correct_shape(self, sample_embeddings):
        """Test that search returns correct shape."""
        store = VectorStore(dimension=384)
        store.add(sample_embeddings)
        
        query = sample_embeddings[:1]  # Use first vector as query
        scores, indices = store.search(query, k=3)
        
        assert scores.shape == (1, 3)
        assert indices.shape == (1, 3)
    
    def test_search_finds_exact_match(self, sample_embeddings):
        """Test that searching with an indexed vector finds itself."""
        store = VectorStore(dimension=384)
        store.add(sample_embeddings)
        
        query = sample_embeddings[:1]
        scores, indices = store.search(query, k=1)
        
        # First result should be index 0 (exact match)
        assert indices[0, 0] == 0
        # Score should be very close to 1 (cosine similarity of identical normalized vectors)
        assert scores[0, 0] > 0.99
    
    def test_save_and_load(self, sample_embeddings):
        """Test saving and loading the index."""
        store = VectorStore(dimension=384)
        store.add(sample_embeddings)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_index.bin")
            
            # Save
            store.save(path)
            assert os.path.exists(path)
            
            # Load into new store
            new_store = VectorStore(dimension=384)
            new_store.load(path)
            
            assert new_store.size == 5
    
    def test_load_nonexistent_raises(self):
        """Test that loading non-existent file raises FileNotFoundError."""
        store = VectorStore(dimension=384)
        
        with pytest.raises(FileNotFoundError):
            store.load("/nonexistent/path/index.bin")
    
    def test_dtype_conversion(self):
        """Test that vectors are converted to float32."""
        store = VectorStore(dimension=384)
        
        # Create float64 vectors
        vectors = np.random.random((3, 384)).astype('float64')
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized = (vectors / norms).astype('float64')  # Still float64
        
        # Should handle conversion internally
        store.add(normalized)
        assert store.size == 3
