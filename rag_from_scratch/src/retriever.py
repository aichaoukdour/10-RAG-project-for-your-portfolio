from typing import List, Dict, Any, Callable
import pandas as pd
from embedding import Embedder
from vector_store import VectorStore
from config import DEFAULT_TOP_K


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        data: pd.DataFrame,
        text_column: str = "text_chunk",
        score_fn: Callable[[str, float, int], float] = None
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.data = data
        self.text_column = text_column
        # Default scoring: use FAISS similarity directly
        self.score_fn = score_fn or (lambda text, base_score, idx: base_score)

    def search(self, query: str, k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
        query_vector = self.embedder.encode([query], show_progress=False)
        scores, indices = self.vector_store.search(query_vector, k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1 or idx >= len(self.data):
                continue
            text = self.data.iloc[idx][self.text_column]
            score = self.score_fn(text, float(scores[0][i]), idx)
            results.append({"text": text, "score": score, "index": int(idx)})

        return results

    def search_with_metadata(
        self,
        query: str,
        k: int = DEFAULT_TOP_K,
        metadata_columns: List[str] = None
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
