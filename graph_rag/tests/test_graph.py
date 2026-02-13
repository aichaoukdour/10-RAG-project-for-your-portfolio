import sys
import os
import unittest

# Add src to the path so imports resolve
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from knowledge_graph import KnowledgeGraph, kg
from retriever import retrieve_graph_context


class TestGraphRAG(unittest.TestCase):

    def setUp(self):
        """Reset the shared graph before each test."""
        kg.clear()
        self.kg_manager = KnowledgeGraph()

    def test_add_triples(self):
        triples = [
            {"head": "A", "relation": "connected_to", "tail": "B"},
            {"head": "B", "relation": "connected_to", "tail": "C"}
        ]
        self.kg_manager.add_triples(triples)
        self.assertEqual(kg.number_of_nodes(), 3)
        self.assertEqual(kg.number_of_edges(), 2)

    def test_retrieval_1_hop(self):
        triples = [
            {"head": "A", "relation": "connected_to", "tail": "B"},
            {"head": "B", "relation": "connected_to", "tail": "C"}
        ]
        self.kg_manager.add_triples(triples)

        context = retrieve_graph_context("A", max_depth=1)
        self.assertIn("A connected_to B", context)
        self.assertNotIn("B connected_to C", context)

    def test_retrieval_2_hop(self):
        triples = [
            {"head": "A", "relation": "connected_to", "tail": "B"},
            {"head": "B", "relation": "connected_to", "tail": "C"}
        ]
        self.kg_manager.add_triples(triples)

        context = retrieve_graph_context("A", max_depth=2)
        self.assertIn("A connected_to B", context)
        self.assertIn("B connected_to C", context)

    def test_retrieval_nonexistent_entity(self):
        context = retrieve_graph_context("Z", max_depth=2)
        self.assertEqual(context, "")

    def test_incoming_edges(self):
        triples = [
            {"head": "X", "relation": "calls", "tail": "Y"},
        ]
        self.kg_manager.add_triples(triples)

        # Querying Y should find the incoming edge from X
        context = retrieve_graph_context("Y", max_depth=1)
        self.assertIn("X calls Y", context)


if __name__ == "__main__":
    unittest.main()
