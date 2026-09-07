import json
import urllib.request
import pandas as pd

GOLD_PATH = "data/processed/gold_healthcare_analytics.csv"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:3b"


def load_healthcare_context():
    df = pd.read_csv(GOLD_PATH)

    context = df[
        [
            "diagnosis",
            "risk_category",
            "patient_count",
            "avg_length_of_stay",
            "avg_total_cost",
            "readmission_rate",
        ]
    ].to_string(index=False)

    return context


def ask_healthcare_ai(question):
    context = load_healthcare_context()

    prompt = f"""
You are a healthcare analytics AI assistant.

Answer the user's question using ONLY the healthcare analytics
data provided below.

Do not invent statistics.
If the answer cannot be determined from the data, say so.
Keep the answer concise and business-friendly.

HEALTHCARE ANALYTICS DATA:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    payload = json.dumps(
        {
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.loads(response.read().decode("utf-8"))

    return result["response"].strip()


if __name__ == "__main__":
    question = "Which patient group has the highest readmission rate?"

    print("\n=== OLLAMA HEALTHCARE AI ASSISTANT ===\n")
    print("Question:", question)
    print("\nLLM Response:")
    print(ask_healthcare_ai(question))
