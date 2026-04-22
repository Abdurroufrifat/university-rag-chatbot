import os
from src.loaders.pdf_loader import load_pdf
from src.loaders.web_loader import load_webpage
from src.processing.cleaner import clean_text
from src.processing.chunker import chunk_text
from src.embeddings.embedder import embed_texts
from src.vectordb.faiss_store import FAISSStore

def ingest_documents(data_folder="data/raw", save_folder="data/processed"):
    all_chunks = []
    all_metadatas = []

    os.makedirs(save_folder, exist_ok=True)

    if not os.path.exists(data_folder):
        print("Data folder not found.")
        return

    # Load PDF files
    for filename in os.listdir(data_folder):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(data_folder, filename)
            pages = load_pdf(file_path)

            for page in pages:
                cleaned = clean_text(page["text"])
                chunks = chunk_text(cleaned, chunk_size=500, overlap=100)

                for chunk in chunks:
                    all_chunks.append(chunk)
                    all_metadatas.append(page["metadata"])

    # Load links from links.txt
    links_file = os.path.join(data_folder, "links.txt")
    if os.path.exists(links_file):
        with open(links_file, "r", encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip()]

        for url in urls:
            docs = load_webpage(url)

            for doc in docs:
                cleaned = clean_text(doc["text"])
                chunks = chunk_text(cleaned, chunk_size=500, overlap=100)

                for chunk in chunks:
                    all_chunks.append(chunk)
                    all_metadatas.append(doc["metadata"])

    if not all_chunks:
        print("No chunks found. Check your PDF files or links.")
        return

    embeddings = embed_texts(all_chunks)
    dim = len(embeddings[0])

    store = FAISSStore(dim)
    store.add(embeddings, all_chunks, all_metadatas)
    store.save(save_folder)

    print(f"Ingestion complete. Stored {len(all_chunks)} chunks.")