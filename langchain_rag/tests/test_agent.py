"""
Tests for the RAG agent routing logic.
"""

import pytest
from src.agent import agent_router


class TestAgentRouter:
    """Tests for the keyword-based intent router."""

    def test_search_route_task_decomposition(self):
        assert agent_router("What is task decomposition?") == "search"

    def test_search_route_agent_topic(self):
        assert agent_router("Tell me about LLM agents") == "search"

    def test_search_route_planning(self):
        assert agent_router("How does planning work?") == "search"

    def test_search_route_chain_of_thought(self):
        assert agent_router("Explain chain of thought prompting") == "search"

    def test_search_route_memory(self):
        assert agent_router("What types of memory are discussed?") == "search"

    def test_direct_route_greeting(self):
        assert agent_router("Hello, how are you?") == "direct"

    def test_direct_route_general(self):
        assert agent_router("What is the capital of France?") == "direct"

    def test_direct_route_math(self):
        assert agent_router("What is 2 + 2?") == "direct"

    def test_case_insensitive(self):
        assert agent_router("TASK DECOMPOSITION methods") == "search"
        assert agent_router("Tell me about MEMORY systems") == "search"
