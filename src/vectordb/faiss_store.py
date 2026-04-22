import faiss
import numpy as np
import pickle
import os

class FAISSStore:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []
        self.metadatas = []

    def add(self, embeddings, texts, metadatas):
        embeddings = np.array(embeddings, dtype="float32")
        self.index.add(embeddings)
        self.texts.extend(texts)
        self.metadatas.extend(metadatas)

    def search(self, query_embedding, k=3):
        query_embedding = np.array([query_embedding], dtype="float32")
        distances, indices = self.index.search(query_embedding, k)

        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.texts):
                results.append({
                    "text": self.texts[idx],
                    "metadata": self.metadatas[idx]
                })
        return results

    def save(self, folder_path):
        os.makedirs(folder_path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(folder_path, "index.faiss"))

        with open(os.path.join(folder_path, "store.pkl"), "wb") as f:
            pickle.dump({
                "texts": self.texts,
                "metadatas": self.metadatas
            }, f)

    def load(self, folder_path):
        self.index = faiss.read_index(os.path.join(folder_path, "index.faiss"))

        with open(os.path.join(folder_path, "store.pkl"), "rb") as f:
            data = pickle.load(f)
            self.texts = data["texts"]
            self.metadatas = data["metadatas"]