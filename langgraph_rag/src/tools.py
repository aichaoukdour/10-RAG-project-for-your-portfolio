from langchain.tools import tool
from .indexer import get_retriever

# Initialize the retriever once
_retriever = None

def _get_or_init_retriever():
    global _retriever
    if _retriever is None:
        _retriever = get_retriever()
    return _retriever

@tool
def retrieve_blog_posts(query: str) -> str:
    """Search and return information about Lilian Weng's blog posts on reward hacking, hallucination, and diffusion video."""
    retriever = _get_or_init_retriever()
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])

# Export the tool
retriever_tool = retrieve_blog_posts
