import networkx as nx
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Module-level directed graph — shared across the application.
# The retrieval function accesses this directly via `kg.nodes`, `kg.successors()`, etc.
kg = nx.DiGraph()


class KnowledgeGraph:
    """Manages the module-level NetworkX knowledge graph."""

    def __init__(self):
        self.graph = kg  # reference the shared module-level graph

    def add_triples(self, triples: List[Dict[str, str]]):
        """Adds a list of (head, relation, tail) triples to the graph."""
        for item in triples:
            try:
                head = item.get("head")
                tail = item.get("tail")
                relation = item.get("relation")

                if head and tail and relation:
                    kg.add_node(head)
                    kg.add_node(tail)
                    kg.add_edge(head, tail, label=relation)
            except Exception as e:
                logger.error(f"Error adding triple {item}: {e}")

    def get_stats(self):
        return {
            "nodes": kg.number_of_nodes(),
            "edges": kg.number_of_edges()
        }
