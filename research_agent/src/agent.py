import time
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import (
    EMBEDDING_MODEL_NAME, 
    PASSAGES_PER_PAGE, 
    TOP_K_PASSAGES, 
    SUMMARY_SENTENCES
)
from .search import search_web, fetch_text
from .text_utils import chunk_passages, split_sentences

class ShortResearchAgent:
    """An agent that automates research by searching, ranking, and summarizing web content."""
    
    def __init__(self, embed_model=EMBEDDING_MODEL_NAME):
        print(f"Initializing Research Agent with model: {embed_model}...")
        self.embedder = SentenceTransformer(embed_model)

    def _cosine_similarity(self, a, b):
        """Compute cosine similarity between two vectors."""
        # Add small epsilon to avoid division by zero
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

    def run(self, query):
        """Execute the research pipeline for a given query."""
        start_time = time.time()
        
        # 1. Search the web
        print(f"Searching for: '{query}'...")
        urls = search_web(query)
        print(f"Found {len(urls)} relevant URLs.")
        
        # 2. Fetch and chunk documents
        docs = []
        for url in urls:
            text = fetch_text(url)
            if not text:
                continue
            
            chunks = chunk_passages(text, max_words=120)
            # Only take the first few passages to keep it focused and fast
            for passage in chunks[:PASSAGES_PER_PAGE]:
                docs.append({"url": url, "passage": passage})
        
        if not docs:
            print("No content could be retrieved from the search results.")
            return {
                "query": query, 
                "passages": [], 
                "summary": "No relevant content found.", 
                "time": time.time() - start_time
            }
            
        # 3. Embed passages and query
        passages_text = [d["passage"] for d in docs]
        passage_embeddings = self.embedder.encode(passages_text, convert_to_numpy=True, show_progress_bar=False)
        query_embedding = self.embedder.encode([query], convert_to_numpy=True, show_progress_bar=False)[0]
        
        # 4. Rank passages by similarity to query
        similarities = [self._cosine_similarity(p_emb, query_embedding) for p_emb in passage_embeddings]
        top_indices = np.argsort(similarities)[::-1][:TOP_K_PASSAGES]
        
        top_passages = [
            {
                "url": docs[i]["url"], 
                "passage": docs[i]["passage"], 
                "score": float(similarities[i])
            } 
            for i in top_indices
        ]
        
        # 5. Generate extractive summary
        summary = self._generate_summary(query_embedding, top_passages)
        
        elapsed_time = time.time() - start_time
        return {
            "query": query,
            "passages": top_passages,
            "summary": summary,
            "time": elapsed_time
        }

    def _generate_summary(self, query_embedding, top_passages):
        """Generate an extractive summary from the top passages."""
        all_sentences = []
        for p in top_passages:
            for sent in split_sentences(p["passage"]):
                all_sentences.append({"text": sent, "url": p["url"]})
        
        if not all_sentences:
            return "Could not generate summary."
            
        sentence_texts = [s["text"] for s in all_sentences]
        sentence_embeddings = self.embedder.encode(sentence_texts, convert_to_numpy=True, show_progress_bar=False)
        
        sentence_sims = [self._cosine_similarity(s_emb, query_embedding) for s_emb in sentence_embeddings]
        top_sent_indices = np.argsort(sentence_sims)[::-1][:SUMMARY_SENTENCES]
        
        chosen_sentences = [all_sentences[idx] for idx in top_sent_indices]
        
        # De-duplicate sentences based on prefix logic
        seen_prefixes = set()
        formatted_lines = []
        for s in chosen_sentences:
            prefix = s["text"].lower()[:80]
            if prefix in seen_prefixes:
                continue
            seen_prefixes.add(prefix)
            formatted_lines.append(f"{s['text']} (Source: {s['url']})")
            
        return " ".join(formatted_lines)
