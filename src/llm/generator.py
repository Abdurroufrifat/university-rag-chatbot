import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.qnaigc.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-v3")

client = None
if API_KEY:
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
    )

def generate_answer(question, retrieved_docs):
    if not retrieved_docs:
        return "I could not find this information in the university documents or website."

    context = "\n\n".join([
        f"Source: {doc['metadata']['source']} | Page: {doc['metadata'].get('page', 'N/A')}\n{doc['text']}"
        for doc in retrieved_docs
    ])

    prompt = f"""
You are a university assistant chatbot.

Answer only from the context below.
If the answer is not in the context, say:
"I could not find this information in the university documents or website."

Context:
{context}

Question:
{question}

Rules:
- Use only the provided context
- Do not make up facts
- Give a short clear answer
- Mention the sources used at the end
"""

    if client is None:
        return build_fallback_answer(retrieved_docs)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content

    except Exception:
        return build_fallback_answer(retrieved_docs)

def build_fallback_answer(retrieved_docs):
    lines = []
    lines.append("LLM service is unavailable right now.")
    lines.append("Here are the most relevant retrieved sources:")

    for i, doc in enumerate(retrieved_docs, start=1):
        source = doc["metadata"]["source"]
        page = doc["metadata"].get("page", "N/A")
        snippet = doc["text"][:300].strip()
        lines.append(f"\n{i}. {source} | Page: {page}\n{snippet}...")

    return "\n".join(lines)