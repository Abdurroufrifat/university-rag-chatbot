from src.embeddings.embedder import embed_query

def retrieve_documents(query, store, k=8):
    query_embedding = embed_query(query)
    results = store.search(query_embedding, k=k)
    return results