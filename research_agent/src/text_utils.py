import re

def chunk_passages(text, max_words=120):
    """Split long text into smaller passages of roughly max_words."""
    words = text.split()
    if not words:
        return []
    
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i : i + max_words])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def split_sentences(text):
    """Split text into sentences using basic regex."""
    # Split on punctuation followed by space
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]
