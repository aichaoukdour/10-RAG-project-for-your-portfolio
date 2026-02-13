from knowledge_graph import kg


# ──────────────────────────────────────────────
# 5. MULTI-HOP RETRIEVAL
# ──────────────────────────────────────────────

def retrieve_graph_context(entity, max_depth=2):
    """
    Retrieves contextual triples from the knowledge graph
    using depth-first multi-hop traversal.

    Traverses both outgoing (successors) and incoming (predecessors)
    edges up to `max_depth` hops from the starting entity.
    """
    context = set()
    visited_nodes = set()

    def dfs(node, depth):
        if depth > max_depth:
            return
        visited_nodes.add(node)

        # 1. Check Outgoing edges (What does this node do?)
        for neighbor in kg.successors(node):
            relation = kg.get_edge_data(node, neighbor)["label"]
            context.add(f"{node} {relation} {neighbor}")
            if neighbor not in visited_nodes:
                dfs(neighbor, depth + 1)

        # 2. Check Incoming edges (Who interacts with this node?)
        for predecessor in kg.predecessors(node):
            relation = kg.get_edge_data(predecessor, node)["label"]
            context.add(f"{predecessor} {relation} {node}")
            if predecessor not in visited_nodes:
                dfs(predecessor, depth + 1)

    if entity in kg.nodes:
        dfs(entity, 1)  # Start the traversal

    return ". ".join(context)
