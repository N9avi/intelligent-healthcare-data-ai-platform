import json
import requests
import pandas as pd
import numpy as np

GOLD_PATH = "data/processed/gold_healthcare_analytics.csv"
OLLAMA_BASE = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:3b"


def get_embedding(text):
    response = requests.post(
        f"{OLLAMA_BASE}/api/embeddings",
        json={
            "model": EMBED_MODEL,
            "prompt": text
        },
        timeout=120
    )
    response.raise_for_status()
    return np.array(response.json()["embedding"], dtype=float)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def build_documents():
    df = pd.read_csv(GOLD_PATH)

    documents = []

    for _, row in df.iterrows():
        text = (
            f"Diagnosis: {row['diagnosis']}. "
            f"Risk category: {row['risk_category']}. "
            f"Patient count: {int(row['patient_count'])}. "
            f"Average age: {row['avg_age']:.2f}. "
            f"Average length of stay: {row['avg_length_of_stay']:.2f} days. "
            f"Average total cost: ${row['avg_total_cost']:.2f}. "
            f"Average previous admissions: {row['avg_previous_admissions']:.2f}. "
            f"Readmission rate: {row['readmission_rate']:.2f}%."
        )

        documents.append(text)

    return documents


def retrieve(question, top_k=5):
    documents = build_documents()
    question_embedding = get_embedding(question)

    scored = []

    for doc in documents:
        doc_embedding = get_embedding(doc)
        score = cosine_similarity(question_embedding, doc_embedding)
        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)

    return [doc for _, doc in scored[:top_k]]


def generate_answer(question, retrieved_docs):
    context = "\n".join(
        f"- {doc}" for doc in retrieved_docs
    )

    prompt = f"""
You are a healthcare analytics assistant.

Use ONLY the retrieved healthcare data below.

Important rules:
- Do not invent numbers.
- If the user asks for highest, lowest, maximum, or minimum,
  compare all retrieved values carefully.
- Include diagnosis, risk category, and the exact percentage
  when answering readmission questions.
- Keep the answer concise.

RETRIEVED HEALTHCARE DATA:
{context}

QUESTION:
{question}

ANSWER:
"""

    response = requests.post(
        f"{OLLAMA_BASE}/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )

    response.raise_for_status()

    return response.json()["response"].strip()


def ask_rag(question):
    docs = retrieve(question)
    answer = generate_answer(question, docs)

    return docs, answer


if __name__ == "__main__":
    question = "Which patient group has the highest readmission rate?"

    print("\n=== HEALTHCARE RAG ASSISTANT ===\n")
    print("Question:", question)

    docs, answer = ask_rag(question)

    print("\nRetrieved Context:")
    for i, doc in enumerate(docs, 1):
        print(f"{i}. {doc}")

    print("\nRAG Answer:")
    print(answer)
