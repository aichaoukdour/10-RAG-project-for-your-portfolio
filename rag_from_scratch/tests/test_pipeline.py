"""
Tests for the pipeline module.
"""
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd

from pipeline import RAGPipeline
from generator import GenerationError, LocalAdvisor
from embedding import Embedder
from vector_store import VectorStore


@pytest.fixture(scope="module")
def pipeline_components():
    """Build real embedder + store with sample data for integration tests."""
    chunks = [
        "In 2024, a Senior-level Data Scientist in US earned 150,000 USD.",
        "In 2024, a Mid-level ML Engineer in US earned 120,000 USD.",
        "In 2023, an Entry-level Data Analyst in UK earned 80,000 USD.",
        "In 2024, a Senior-level Data Engineer in Germany earned 130,000 USD.",
        "In 2023, a Mid-level AI Researcher in US earned 160,000 USD.",
    ]
    df = pd.DataFrame({"text_chunk": chunks})
    embedder = Embedder()
    store = VectorStore(dimension=embedder.dimension)
    vectors = embedder.encode(chunks)
    store.add(vectors)
    return embedder, store, df


class TestRAGPipelineRun:
    """Tests for the RAGPipeline.run() method."""

    def test_run_returns_expected_keys(self, pipeline_components):
        """Test that run() returns the correct response structure."""
        embedder, store, df = pipeline_components

        with patch("pipeline.Generator") as MockGen:
            mock_gen = MockGen.return_value
            mock_gen.generate_answer.return_value = "The salary is $150,000."

            pipeline = RAGPipeline.__new__(RAGPipeline)
            pipeline.retriever = __import__("retriever").Retriever(embedder, store, df)
            pipeline.generator = mock_gen
            pipeline.fallback = LocalAdvisor()
            pipeline.model = "test-model"

            result = pipeline.run("Data Scientist salary")

        assert "query" in result
        assert "answer" in result
        assert "context" in result
        assert "scores" in result
        assert "source" in result

    def test_run_uses_fallback_on_generation_error(self, pipeline_components):
        """Test that the fallback is used when the generator raises."""
        embedder, store, df = pipeline_components

        with patch("pipeline.Generator") as MockGen:
            mock_gen = MockGen.return_value
            mock_gen.generate_answer.side_effect = GenerationError("API down")

            pipeline = RAGPipeline.__new__(RAGPipeline)
            pipeline.retriever = __import__("retriever").Retriever(embedder, store, df)
            pipeline.generator = mock_gen
            pipeline.fallback = LocalAdvisor()
            pipeline.model = "test-model"

            result = pipeline.run("Data Scientist salary", use_fallback=True)

        assert result["source"] == "fallback"
        assert "LocalAdvisor" in result["answer"]

    def test_run_raises_when_fallback_disabled(self, pipeline_components):
        """Test that GenerationError propagates when fallback is disabled."""
        embedder, store, df = pipeline_components

        with patch("pipeline.Generator") as MockGen:
            mock_gen = MockGen.return_value
            mock_gen.generate_answer.side_effect = GenerationError("API down")

            pipeline = RAGPipeline.__new__(RAGPipeline)
            pipeline.retriever = __import__("retriever").Retriever(embedder, store, df)
            pipeline.generator = mock_gen
            pipeline.fallback = LocalAdvisor()
            pipeline.model = "test-model"

            with pytest.raises(GenerationError):
                pipeline.run("Data Scientist salary", use_fallback=False)

    def test_run_returns_no_results(self, pipeline_components):
        """Test that a completely irrelevant query returns the no_results response."""
        embedder, store, df = pipeline_components

        with patch("pipeline.Generator") as MockGen:
            mock_gen = MockGen.return_value

            pipeline = RAGPipeline.__new__(RAGPipeline)
            # Use a retriever with a very high threshold to force no results
            retriever = __import__("retriever").Retriever(
                embedder, store, df, min_score=0.99
            )
            pipeline.retriever = retriever
            pipeline.generator = mock_gen
            pipeline.fallback = LocalAdvisor()
            pipeline.model = "test-model"

            result = pipeline.run("xyzzy foobar nonsense gibberish")

        assert result["source"] == "no_results"
        assert len(result["context"]) == 0


class TestSalaryInsight:
    """Tests for the get_salary_insight() method."""

    def test_insight_uses_retriever_not_run(self, pipeline_components):
        """Verify that get_salary_insight calls retriever.search, not self.run."""
        embedder, store, df = pipeline_components

        with patch("pipeline.Generator") as MockGen:
            mock_gen = MockGen.return_value
            mock_gen.generate_answer.return_value = "Insight report text."

            pipeline = RAGPipeline.__new__(RAGPipeline)
            pipeline.retriever = __import__("retriever").Retriever(embedder, store, df)
            pipeline.generator = mock_gen
            pipeline.fallback = LocalAdvisor()
            pipeline.model = "test-model"

            result = pipeline.get_salary_insight("Data Scientist")

        assert "job_title" in result
        assert "report" in result
        assert "num_records_analyzed" in result
        # The generator should only be called ONCE (for the insight), not twice
        assert mock_gen.generate_answer.call_count == 1

    def test_insight_fallback_on_error(self, pipeline_components):
        """Test that insight falls back to local report on GenerationError."""
        embedder, store, df = pipeline_components

        with patch("pipeline.Generator") as MockGen:
            mock_gen = MockGen.return_value
            mock_gen.generate_answer.side_effect = GenerationError("quota")

            pipeline = RAGPipeline.__new__(RAGPipeline)
            pipeline.retriever = __import__("retriever").Retriever(embedder, store, df)
            pipeline.generator = mock_gen
            pipeline.fallback = LocalAdvisor()
            pipeline.model = "test-model"

            result = pipeline.get_salary_insight("Data Scientist")

        assert "Career Insight Report" in result["report"]
        assert "local report" in result["report"].lower()
