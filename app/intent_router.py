
from ollama import Client
from app.config import OLLAMA_MODEL

client = Client(host="http://localhost:11434")


def classify_intent(question: str) -> str:

    prompt = f"""
You are an enterprise AI supervisor.

Classify the user's question into exactly ONE category:

RAG:
Questions requiring information from enterprise documents,
policies, procedures, guidelines or contracts.

SQL:
Questions requiring structured business data from Snowflake,
such as employee details, salaries, purchase orders or invoices.

BOTH:
Questions requiring BOTH enterprise document knowledge
and structured business data.

UNKNOWN:
Questions outside enterprise HR, Procurement and Finance.

Examples:

Question: What is the employee leave policy?
Category: RAG

Question: How many employees are in the HR department?
Category: SQL

Question: What is the invoice approval policy and how many invoices are pending?
Category: BOTH

Question: What is the capital of France?
Category: UNKNOWN

Return ONLY one category:
RAG, SQL, BOTH or UNKNOWN.

User question: {question}

Category:
"""

    response = client.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={"temperature": 0}
    )

    intent = response.message.content.strip().upper()

    allowed_intents = {"RAG", "SQL", "BOTH", "UNKNOWN"}

    # Extract a valid category even if the model adds extra text.
    for category in allowed_intents:
        if intent == category:
            return category

    for line in intent.splitlines():
        cleaned = line.strip(" .,:*`")
        if cleaned in allowed_intents:
            return cleaned

    return "UNKNOWN"


if __name__ == "__main__":

    print("ENTERPRISE GENAI INTENT CLASSIFIER")
    print("---------------------------------")

    while True:

        question = input("\nAsk your business question: ").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        intent = classify_intent(question)

        print(f"\nDetected Intent: {intent}")
