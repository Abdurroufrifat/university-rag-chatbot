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

st.set_page_config(
    page_title="University Chatbot",
    page_icon="🎓",
    layout="wide"
)

INDEX_PATH = "data/processed/index.faiss"
STORE_PATH = "data/processed/store.pkl"
RAW_DATA_PATH = "data/raw"

def ensure_index_exists():
    os.makedirs(RAW_DATA_PATH, exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    if not os.path.exists(INDEX_PATH) or not os.path.exists(STORE_PATH):
        with st.spinner("Building knowledge base for the first time..."):
            ingest_documents()

@st.cache_resource
def load_store():
    ensure_index_exists()
    store = FAISSStore(dim=384)
    store.load("data/processed")
    return store

def reload_store():
    st.cache_resource.clear()
    return load_store()

def save_uploaded_files(uploaded_files):
    os.makedirs(RAW_DATA_PATH, exist_ok=True)
    saved_files = []

    for uploaded_file in uploaded_files:
        save_path = os.path.join(RAW_DATA_PATH, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.read())
        saved_files.append(uploaded_file.name)

    return saved_files

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")

    language_mode = st.selectbox(
        "Answer Language",
        ["Auto", "English", "Bangla"]
    )

    st.markdown("---")
    st.subheader("📄 Upload PDFs")

    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        saved_files = save_uploaded_files(uploaded_files)
        st.success(f"Uploaded: {', '.join(saved_files)}")

    if st.button("Rebuild Knowledge Base"):
        with st.spinner("Rebuilding knowledge base..."):
            ingest_documents()
            reload_store()
        st.success("Knowledge base rebuilt successfully.")

    st.markdown("---")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.info(
        "This chatbot answers from uploaded university PDFs and website links."
    )

# Main title
st.title("🎓 University Chatbot")
st.markdown("Ask questions from official university PDFs and website pages.")

# Load FAISS store
store = load_store()

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show old chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("View Sources"):
                for i, doc in enumerate(msg["sources"], start=1):
                    source = doc["metadata"]["source"]
                    page = doc["metadata"].get("page", "N/A")
                    source_type = doc["metadata"].get("type", "unknown")

                    if source_type == "web":
                        st.markdown(f"**{i}.** {source}")
                    else:
                        st.markdown(f"**{i}.** {source} (Page {page})")

                    with st.expander(f"Show chunk {i}"):
                        st.write(doc["text"])

# Chat input
user_input = st.chat_input("Ask your question")

if user_input:
    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching and generating answer..."):
            retrieved_docs = retrieve_documents(user_input, store, k=8)
            answer = generate_answer(user_input, retrieved_docs, language_mode=language_mode)

        st.write(answer)

        with st.expander("View Sources"):
            for i, doc in enumerate(retrieved_docs[:3], start=1):
                source = doc["metadata"]["source"]
                page = doc["metadata"].get("page", "N/A")
                source_type = doc["metadata"].get("type", "unknown")

                if source_type == "web":
                    st.markdown(f"**{i}.** {source}")
                else:
                    st.markdown(f"**{i}.** {source} (Page {page})")

                with st.expander(f"Show chunk {i}"):
                    st.write(doc["text"])

    # Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": retrieved_docs[:3]
    })