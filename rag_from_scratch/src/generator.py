"""
Generator module for LLM-based answer generation.
Supports OpenAI API with automatic fallback to local rule-based advisor.
"""
import logging
from typing import List, Optional

from openai import OpenAI, APIConnectionError, RateLimitError, AuthenticationError

from config import (
    OPENAI_API_KEY, OPENAI_BASE_URL,
    DEFAULT_LLM_MODEL, LLM_TEMPERATURE
)

# Setup module logger
# Setup module logger
logger = logging.getLogger(__name__)


class Generator:
    """
    LLM-based answer generator using OpenAI's API.
    
    Generates grounded answers based only on provided context,
    following RAG best practices.
    """
    
    def __init__(
        self, 
        model: str = DEFAULT_LLM_MODEL, 
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> None:
        """
        Initialize the Generator with an LLM client.
        
        Args:
            model: Name of the LLM model to use.
            api_key: OpenAI API key (falls back to env var if not provided).
            base_url: Optional base URL for OpenAI-compatible APIs (e.g., Ollama).
        """
        self.model = model
        self.api_key = api_key or OPENAI_API_KEY
        self.base_url = base_url or OPENAI_BASE_URL
        
        # Initialize client (works for OpenAI-compatible APIs like Ollama/vLLM too)
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
            logger.info(f"Using custom API endpoint: {self.base_url}")
            
        self.client = OpenAI(**client_kwargs)
        logger.info(f"Generator initialized with model: {model}")

    def generate_answer(
        self, 
        query: str, 
        context_chunks: List[str],
        temperature: float = LLM_TEMPERATURE
    ) -> str:
        """
        Generate an answer based ONLY on the provided context.
        
        Args:
            query: The user's question.
            context_chunks: List of relevant text chunks from retrieval.
            temperature: LLM temperature (0 for deterministic output).
            
        Returns:
            Generated answer string, or error message if generation fails.
        """
        if not self.api_key:
            logger.error("OPENAI_API_KEY is missing")
            return "Error: OPENAI_API_KEY is missing. Please set it in your .env file."

        if not context_chunks:
            logger.warning("No context provided for answer generation")
            return "I don't have enough data to answer that question."

        context_text = "\n---\n".join(context_chunks)
        
        prompt = f"""Answer the question ONLY using the provided context.
If the answer is not in the context, say: 'I don't have enough data to answer that.'

CONTEXT:
{context_text}

QUESTION:
{query}

ANSWER:"""

        try:
            logger.debug(f"Sending request to LLM with {len(context_chunks)} context chunks")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional AI Career Advisor."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )
            answer = response.choices[0].message.content.strip()
            logger.info(f"Generated answer ({len(answer)} chars)")
            return answer
            
        except AuthenticationError:
            logger.error("Invalid OpenAI API key.")
            return "Error: Invalid OpenAI API key. Please check your .env file."
        except RateLimitError:
            logger.error("OpenAI rate limit reached.")
            return "Error: OpenAI rate limit reached. Please wait a moment and try again."
        except APIConnectionError as e:
            logger.error(f"Failed to connect to OpenAI: {e}")
            return "Error: Failed to connect to OpenAI API. Please check your internet connection."
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return f"Error generating answer: {e}"


class LocalAdvisor:
    """
    A local, rule-based fallback that doesn't require an LLM.
    
    Perfect for demonstrating the RAG pipeline when API credits 
    are unavailable or for offline usage.
    """
    
    def __init__(self) -> None:
        """Initialize the LocalAdvisor."""
        logger.info("LocalAdvisor fallback initialized")
    
    def generate_answer(
        self, 
        query: str, 
        context_chunks: List[str],
        max_chunks: int = 3
    ) -> str:
        """
        Generate a simple synthesis of the retrieved context.
        
        Args:
            query: The user's question (used for logging).
            context_chunks: List of relevant text chunks.
            max_chunks: Maximum number of chunks to include in response.
            
        Returns:
            Formatted string summarizing the retrieved context.
        """
        logger.info(f"LocalAdvisor processing query with {len(context_chunks)} chunks")
        
        if not context_chunks:
            return "I don't have enough data in my local knowledge base to answer that."
        
        # Simple rule-based 'synthesis' of the context
        summary = "Based on my semantic search, here are the key facts:\n"
        for i, chunk in enumerate(context_chunks[:max_chunks]):
            summary += f"• {chunk}\n"
        
        summary += "\n[Note: This answer was generated by the LocalAdvisor fallback because the LLM API is unavailable.]"
        return summary


if __name__ == "__main__":
    # Test LocalAdvisor (doesn't require API)
    advisor = LocalAdvisor()
    mock_context = [
        "In 2024, a Data Scientist in New York earns $150,000.",
        "Senior Data Scientists can earn up to $200,000."
    ]
    answer = advisor.generate_answer("How much does a Data Scientist make?", mock_context)
    print(f"LocalAdvisor Answer:\n{answer}")
