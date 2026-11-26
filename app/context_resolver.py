
from ollama import Client
from app.config import OLLAMA_MODEL

client = Client(host="http://localhost:11434")


def resolve_context(question: str, history: list) -> str:

    if not history:
        return question

    recent_history = history[-6:]

    conversation = ""

    for message in recent_history:
        role = message.get("role", "user").upper()
        content = message.get("content", "")
        conversation += f"{role}: {content}\n"

    prompt = f"""
You are an enterprise conversation context resolver.

Your task is to rewrite the latest user question into a
standalone question using the conversation history.

Rules:
1. Preserve the user's original intent.
2. Resolve references such as "their", "those", "it" and "them".
3. Do not answer the question.
4. Do not invent business facts.
5. If the question is already standalone, return it unchanged.
6. Return only the rewritten question.

Conversation:
{conversation}

Latest question:
{question}

Standalone question:
"""

    response = client.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        options={"temperature": 0}
    )

    resolved = response.message.content.strip()

    return resolved if resolved else question
