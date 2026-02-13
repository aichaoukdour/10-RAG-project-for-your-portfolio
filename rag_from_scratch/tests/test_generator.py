"""
Tests for the generator module.
"""
import pytest

from generator import BaseGenerator, Generator, GenerationError, LocalAdvisor


class TestLocalAdvisor:
    """Tests for the LocalAdvisor fallback generator."""

    @pytest.fixture
    def advisor(self):
        return LocalAdvisor()

    def test_returns_string(self, advisor):
        """Test that generate_answer returns a string."""
        chunks = ["Data Scientist earns $150,000 in US."]
        answer = advisor.generate_answer("salary?", chunks)
        assert isinstance(answer, str)

    def test_empty_context_returns_fallback_message(self, advisor):
        """Test that empty context returns a clear no-data message."""
        answer = advisor.generate_answer("salary?", [])
        assert "don't have enough data" in answer.lower()

    def test_includes_context_chunks(self, advisor):
        """Test that the answer includes the provided context chunks."""
        chunks = [
            "Data Scientist in New York earns $150,000.",
            "ML Engineer in SF earns $180,000."
        ]
        answer = advisor.generate_answer("salary?", chunks)
        assert "150,000" in answer
        assert "180,000" in answer

    def test_respects_max_chunks(self, advisor):
        """Test that max_chunks limits the number of chunks in the answer."""
        chunks = [f"Chunk {i}" for i in range(10)]
        answer = advisor.generate_answer("query", chunks, max_chunks=2)
        assert "Chunk 0" in answer
        assert "Chunk 1" in answer
        assert "Chunk 2" not in answer

    def test_includes_fallback_note(self, advisor):
        """Test that the answer includes the LocalAdvisor attribution."""
        chunks = ["Some data."]
        answer = advisor.generate_answer("query", chunks)
        assert "LocalAdvisor" in answer


class TestBaseGenerator:
    """Tests for the abstract BaseGenerator class."""

    def test_cannot_instantiate(self):
        """Test that BaseGenerator cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseGenerator()

    def test_subclass_must_implement_generate_answer(self):
        """Test that subclasses without generate_answer raise TypeError."""
        class IncompleteGenerator(BaseGenerator):
            pass

        with pytest.raises(TypeError):
            IncompleteGenerator()


class TestGeneratorInit:
    """Tests for the Generator facade initialization."""

    def test_missing_api_key_raises(self, monkeypatch):
        """Test that missing API key raises ValueError."""
        monkeypatch.setattr("generator.GEMINI_API_KEY", None)
        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            Generator(api_key=None)


class TestGenerationError:
    """Tests for the GenerationError exception."""

    def test_is_exception(self):
        assert issubclass(GenerationError, Exception)

    def test_message(self):
        err = GenerationError("test failure")
        assert "test failure" in str(err)
