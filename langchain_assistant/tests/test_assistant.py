import pytest
from unittest.mock import patch, MagicMock
from src.assistant import get_assistant_chain

def test_chain_initialization():
    """Verify that the chain can be initialized."""
    with patch('src.assistant.OllamaLLM'), patch('src.assistant.DuckDuckGoSearchRun'):
        chain = get_assistant_chain()
        assert chain is not None

def test_chain_execution():
    """Verify the chain invokes the search tool and LLM correctly."""
    with patch('src.assistant.OllamaLLM') as mock_llm_class, \
         patch('src.assistant.DuckDuckGoSearchRun') as mock_search_class:
        
        # Setup mock instance for LLM
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm
        # Crucial: set the return value for the invoke call
        mock_llm.invoke.return_value = "Final Answer"
        
        # Setup mock instance for Search
        mock_search = MagicMock()
        mock_search_class.return_value = mock_search
        mock_search.run.return_value = "Search context"
        
        # Get chain
        chain = get_assistant_chain()
        
        # Invoke chain
        response = chain.invoke({"question": "test question"})
        
        # Assertions
        assert str(response) == "Final Answer"
        assert mock_llm.invoke.called
        assert mock_search.run.called
