import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from src.nodes import (
    generate_query_or_respond, 
    grade_documents, 
    rewrite_question, 
    generate_answer
)

def test_rewrite_question():
    """Verify rewrite_question node works as expected."""
    with patch('src.nodes.model') as mock_model:
        mock_model.invoke.return_value = AIMessage(content="Rewritten question")
        
        state = {"messages": [HumanMessage(content="old q")]}
        result = rewrite_question(state)
        
        assert "messages" in result
        assert isinstance(result["messages"][0], HumanMessage)
        assert result["messages"][0].content == "Rewritten question"

def test_grade_documents_relevant():
    """Verify grade_documents returns generate_answer for relevant docs."""
    with patch('src.nodes.model') as mock_model:
        mock_grader = MagicMock()
        mock_grader.invoke.return_value = MagicMock(binary_score="yes")
        mock_model.with_structured_output.return_value = mock_grader
        
        state = {
            "messages": [
                HumanMessage(content="test q"),
                ToolMessage(content="relevant doc content", tool_call_id="1")
            ]
        }
        
        result = grade_documents(state)
        assert result == "generate_answer"

def test_grade_documents_irrelevant():
    """Verify grade_documents returns rewrite_question for irrelevant docs."""
    with patch('src.nodes.model') as mock_model:
        mock_grader = MagicMock()
        mock_grader.invoke.return_value = MagicMock(binary_score="no")
        mock_model.with_structured_output.return_value = mock_grader
        
        state = {
            "messages": [
                HumanMessage(content="test q"),
                ToolMessage(content="meow", tool_call_id="1")
            ]
        }
        
        result = grade_documents(state)
        assert result == "rewrite_question"

def test_generate_answer():
    """Verify generate_answer node works as expected."""
    with patch('src.nodes.model') as mock_model:
        mock_model.invoke.return_value = AIMessage(content="Final Answer")
        
        state = {
            "messages": [
                HumanMessage(content="test q"),
                ToolMessage(content="context content", tool_call_id="1")
            ]
        }
        
        result = generate_answer(state)
        assert result["messages"][0].content == "Final Answer"
