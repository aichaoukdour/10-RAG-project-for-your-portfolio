import pytest
from unittest.mock import patch, MagicMock
from src.agent import ShortResearchAgent

@pytest.fixture
def agent():
    # Mock SentenceTransformer to avoid downloading model and slow tests
    with patch('src.agent.SentenceTransformer') as mock_st:
        # Mock encoder to return dummy numpy arrays
        mock_model = MagicMock()
        mock_model.encode.side_effect = lambda texts, **kwargs: [eval("0.1") * i for i in range(len(texts))] if isinstance(texts, list) else 0.1
        # In actual code it returns numpy arrays, but for simplicity of mock logic:
        mock_model.encode.return_value = MagicMock() 
        mock_st.return_value = mock_model
        return ShortResearchAgent()

def test_agent_initialization(agent):
    assert agent.embedder is not None

@patch('src.agent.search_web')
@patch('src.agent.fetch_text')
def test_agent_run_no_results(mock_fetch, mock_search, agent):
    mock_search.return_value = []
    
    result = agent.run("test query")
    assert result["query"] == "test query"
    assert result["passages"] == []
    assert result["summary"] == "No relevant content found."

@patch('src.agent.search_web')
@patch('src.agent.fetch_text')
@patch.object(ShortResearchAgent, '_generate_summary')
def test_agent_run_with_results(mock_summary, mock_fetch, mock_search, agent):
    mock_search.return_value = ["http://example.com"]
    mock_fetch.return_value = "This is a test passage that is long enough to be chunked."
    mock_summary.return_value = "Mock summary"
    
    # Mock embedder encode behaviors more specifically
    agent.embedder.encode.side_effect = [
        [[0.1]], # passage embeddings (list of 1)
        [0.1]    # query embedding
    ]
    
    # Mock cosine similarity to return 1.0 (perfect match)
    with patch.object(ShortResearchAgent, '_cosine_similarity', return_value=1.0):
        result = agent.run("test query")
        
        assert len(result["passages"]) > 0
        assert result["summary"] == "Mock summary"
        assert result["passages"][0]["url"] == "http://example.com"
