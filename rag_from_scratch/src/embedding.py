from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME


class Embedder:

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME, device: str | None = None) -> None:
        self.model = SentenceTransformer(model_name, device=device)

    def encode(
        self,
        texts: Union[str, List[str]],
        show_progress: bool = True,
        normalize: bool = True,
        batch_size: int = 32
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            show_progress_bar=show_progress,
            batch_size=batch_size
        )

        embeddings = np.array(embeddings, dtype="float32", copy=False)

        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            embeddings /= norms

        return embeddings

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()


if __name__ == "__main__":
    embedder = Embedder()
    test_texts = ["This is a test sentence.", "Another test sentence."]
    vectors = embedder.encode(test_texts, show_progress=False)
    print(f"Vectors shape: {vectors.shape}, Dimension: {embedder.dimension}")
