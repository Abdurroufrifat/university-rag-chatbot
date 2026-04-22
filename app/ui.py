import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.pipeline.ingest import ingest_documents
from src.vectordb.faiss_store import FAISSStore
from src.retrieval.retriever import retrieve_documents
from src.llm.generator import generate_answer

st.set_page_config(page_title="University Chatbot", page_icon="🎓", layout="wide")

st.title("🎓 University Chatbot")
st.markdown("Ask questions from official university PDFs and website pages.")

INDEX_PATH = "data/processed/index.faiss"
STORE_PATH = "data/processed/store.pkl"

def ensure_index_exists():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(STORE_PATH):
        with st.spinner("Building knowledge base for the first time..."):
            ingest_documents()

@st.cache_resource
def load_store():
    ensure_index_exists()
    store = FAISSStore(dim=384)
    store.load("data/processed")
    return store

store = load_store()

question = st.text_input("Ask your question")

if st.button("Get Answer"):
    if question.strip():
        retrieved_docs = retrieve_documents(question, store, k=3)
        answer = generate_answer(question, retrieved_docs)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources")
        for i, doc in enumerate(retrieved_docs, start=1):
            source = doc["metadata"]["source"]
            page = doc["metadata"].get("page", "N/A")
            source_type = doc["metadata"].get("type", "unknown")

            if source_type == "web":
                st.markdown(f"**{i}.** {source}")
            else:
                st.markdown(f"**{i}.** {source} (Page {page})")

            with st.expander(f"Show chunk {i}"):
                st.write(doc["text"])
    else:
        st.warning("Please enter a question.")