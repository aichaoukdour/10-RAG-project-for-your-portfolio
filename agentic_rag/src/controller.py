from src.config import SEARCH_KEYWORDS

def agent_controller(query: str) -> str:
    """
    Decides whether to 'search' the knowledge base or answer 'direct'.
    Uses keyword-based routing to identify intent.
    """
    q_lower = query.lower()
    if any(word in q_lower for word in SEARCH_KEYWORDS):
        return "search"
    return "direct"
