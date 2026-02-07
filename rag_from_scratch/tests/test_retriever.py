"""
Tests for the retriever module.
"""
import pytest
import pandas as pd
import numpy as np

from retriever import Retriever
from embedding import Embedder
from vector_store import VectorStore


class TestRetriever:
    """Tests for the Retriever class."""
    
    @pytest.fixture(scope="class")
    def setup_retriever(self, sample_text_chunks):
        """Create a retriever with sample data."""
        embedder = Embedder()
        
        # Create embeddings for sample chunks
        vectors = embedder.encode(sample_text_chunks)
        
        # Setup vector store
        store = VectorStore(dimension=embedder.dimension)
        store.add(vectors)
        
        # Create DataFrame
        df = pd.DataFrame({'text_chunk': sample_text_chunks})
        
        return Retriever(embedder, store, df)
    
    @pytest.fixture
    def retriever_with_data(self):
        """Create a fresh retriever for each test."""
        chunks = [
            "Data Scientist in New York earns $150,000 USD.",
            "Machine Learning Engineer in San Francisco earns $180,000 USD.",
            "Data Analyst in Chicago earns $80,000 USD.",
            "Senior Data Engineer in Seattle earns $160,000 USD.",
            "Entry-level Data Scientist earns $90,000 USD."
        ]
        
        embedder = Embedder()
        vectors = embedder.encode(chunks)
        
        store = VectorStore(dimension=embedder.dimension)
        store.add(vectors)
        
        df = pd.DataFrame({'text_chunk': chunks})
        
        return Retriever(embedder, store, df)
    
    def test_initialization(self, retriever_with_data):
        """Test that retriever initializes correctly."""
        assert retriever_with_data.embedder is not None
        assert retriever_with_data.vector_store is not None
        assert len(retriever_with_data.data) == 5
    
    def test_search_returns_results(self, retriever_with_data):
        """Test that search returns results."""
        results = retriever_with_data.search("Data Scientist salary", k=3)
        
        assert len(results) > 0
        assert len(results) <= 3
    
    def test_search_result_structure(self, retriever_with_data):
        """Test that search results have correct structure."""
        results = retriever_with_data.search("Data Scientist", k=1)
        
        assert len(results) == 1
        result = results[0]
        
        assert 'text' in result
        assert 'score' in result
        assert 'index' in result
        assert isinstance(result['score'], float)
        assert isinstance(result['index'], int)
    
    def test_search_relevance(self, retriever_with_data):
        """Test that search returns relevant results."""
        results = retriever_with_data.search("Machine Learning Engineer", k=3)
        
        # The ML Engineer chunk should be in top results
        texts = [r['text'] for r in results]
        ml_found = any('Machine Learning' in t for t in texts)
        
        assert ml_found, "Expected ML Engineer in search results"
    
    def test_search_scores_are_ordered(self, retriever_with_data):
        """Test that search results are ordered by score (descending)."""
        results = retriever_with_data.search("Data Scientist salary", k=5)
        
        scores = [r['score'] for r in results]
        assert scores == sorted(scores, reverse=True)
    
    def test_search_with_k_greater_than_data(self, retriever_with_data):
        """Test search when k is larger than number of documents."""
        results = retriever_with_data.search("salary", k=100)
        
        # Should return at most 5 (number of documents)
        assert len(results) <= 5
