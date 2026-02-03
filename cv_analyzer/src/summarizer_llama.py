from llm_client import query_llm

def summarize_cv_llama(text: str) -> str:
    prompt = f"""
    Summarize the following CV into 3-5 concise bullet points highlighting
    key skills, experiences, and achievements:

    {text}

    Only return bullet points.
    """
    return query_llm(prompt, max_tokens=1024)

