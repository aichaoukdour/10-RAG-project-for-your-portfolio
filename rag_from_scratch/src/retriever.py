import logging
from typing import List, Dict, Any, Callable, Optional

import pandas as pd
from embedding import Embedder
from vector_store import VectorStore
from config import DEFAULT_TOP_K

logger = logging.getLogger(__name__)

# Minimum cosine similarity to consider a chunk relevant
MIN_SCORE_THRESHOLD = 0.25


class Retriever:
    """Performs semantic search over stored document chunks.

    Attributes:
        embedder: Encodes queries into vectors.
        vector_store: FAISS index for similarity search.
        data: DataFrame containing the text chunks.
        text_column: Name of the column holding chunk text.
        score_fn: Optional re-scoring function with signature
            ``(text: str, base_score: float, row_index: int) -> float``.
            Defaults to returning the base similarity score unchanged.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        data: pd.DataFrame,
        text_column: str = "text_chunk",
        score_fn: Optional[Callable[[str, float, int], float]] = None,
        min_score: float = MIN_SCORE_THRESHOLD,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.data = data
        self.text_column = text_column
        self.score_fn = score_fn or (lambda text, base_score, idx: base_score)
        self.min_score = min_score

    def search(self, query: str, k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
        """Return up to *k* chunks whose similarity exceeds ``min_score``."""
        query_vector = self.embedder.encode([query], show_progress=False)
        scores, indices = self.vector_store.search(query_vector, k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1 or idx >= len(self.data):
                continue
            text = self.data.iloc[idx][self.text_column]
            score = self.score_fn(text, float(scores[0][i]), idx)
            if score < self.min_score:
                continue
            results.append({"text": text, "score": score, "index": int(idx)})

        return results

    def search_with_metadata(
        self,
        query: str,
        k: int = DEFAULT_TOP_K,
        metadata_columns: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        results = self.search(query, k)

        if metadata_columns:
            for result in results:
                idx = result["index"]
                for col in metadata_columns:
                    if col in self.data.columns and col != self.text_column:
                        result[col] = self.data.iloc[idx][col]

        return results


if __name__ == "__main__":
    print("Retriever module with custom scoring loaded successfully")
