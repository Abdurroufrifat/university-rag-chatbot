import os
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI

load_dotenv()

def get_secret(name, default=None):
    if name in st.secrets:
        return st.secrets[name]
    return os.getenv(name, default)

API_KEY = get_secret("OPENAI_API_KEY")
BASE_URL = get_secret("OPENAI_BASE_URL", "https://api.qnaigc.com/v1")
MODEL_NAME = get_secret("MODEL_NAME", "deepseek-v3")

client = None
if API_KEY:
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
    )

def build_fallback_answer(retrieved_docs):
    if not retrieved_docs:
        return "I could not find this information in the university documents or website."

    lines = []
    lines.append("LLM service is unavailable right now. Here are the most relevant retrieved sources:")

    for i, doc in enumerate(retrieved_docs[:3], start=1):
        source = doc["metadata"]["source"]
        page = doc["metadata"].get("page", "N/A")
        snippet = doc["text"][:350].strip()
        lines.append(f"\n{i}. {source} | Page: {page}\n{snippet}...")

    return "\n".join(lines)

def generate_answer(question, retrieved_docs, language_mode="Auto"):
    if not retrieved_docs:
        return "I could not find this information in the university documents or website."

    # Use top 3 from retrieved docs for final answer generation
    top_docs = retrieved_docs[:3]

    context = "\n\n".join([
        f"Source: {doc['metadata']['source']} | Page: {doc['metadata'].get('page', 'N/A')}\n{doc['text']}"
        for doc in top_docs
    ])

    language_instruction = ""
    if language_mode == "Bangla":
        language_instruction = "Answer in Bangla."
    elif language_mode == "English":
        language_instruction = "Answer in English."
    else:
        language_instruction = """
If the question is in Bangla, answer in Bangla.
If the question is in English, answer in English.
"""

    prompt = f"""
You are a university assistant chatbot.

Answer only from the context below.
If the answer is not in the context, say:
"I could not find this information in the university documents or website."

{language_instruction}

Context:
{context}

Question:
{question}

Rules:
- Use only the provided context
- Do not make up facts
- Keep the answer clear and concise
- Mention the sources used at the end
"""

    if client is None:
        return build_fallback_answer(top_docs)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception as e:
        return build_fallback_answer(top_docs) + f"\n\nAPI Error: {str(e)}"