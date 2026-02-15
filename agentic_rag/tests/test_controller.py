import pytest
from src.controller import agent_controller

@pytest.mark.parametrize("query,expected", [
    ("summarize the pdf", "search"),
    ("find info in document", "search"),
    ("what is help?", "direct"),
    ("hello world", "direct"),
    ("extract data from file", "search"),
    ("who are you?", "direct"),
])
def test_agent_controller_routing(query, expected):
    """Test if the controller routes correctly based on keywords."""
    assert agent_controller(query) == expected
