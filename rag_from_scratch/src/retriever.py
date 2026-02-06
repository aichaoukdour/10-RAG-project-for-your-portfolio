import numpy as np

class Retriever:
    def __init__(self, embedder, vector_store, data):
        """
        Initialize the retriever.
        :param embedder: Instance of Embedder class.
        :param vector_store: Instance of VectorStore class.
        :param data: Pandas DataFrame containing the text chunks.
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.data = data

    def search(self, query, k=5):
        """
        Search for the top k most similar records to the query.
        """
        # 1. Encode query
        query_vector = self.embedder.encode([query])
        
        # 2. Search FAISS index
        # IndexFlatL2 return (distances, indices)
        distances, indices = self.vector_store.index.search(query_vector, k)
        
        # 3. Retrieve results from data
        results = []
        for idx in indices[0]:
            if idx != -1: # FAISS returns -1 if not enough results
                results.append(self.data.iloc[idx]['text_chunk'])
        
        return results

if __name__ == "__main__":
    # Test would go here, but we need the components initialized
    pass
