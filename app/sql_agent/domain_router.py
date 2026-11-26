
from ollama import Client
from app.config import OLLAMA_MODEL

client = Client(host="http://localhost:11434")


def detect_domain(question: str) -> str:

    prompt = f"""
You are an Enterprise Business Domain Classifier.

Classify the user's question into exactly one category:

HR
PROCUREMENT
FINANCE
UNKNOWN

Classification rules:

HR:
Employee information, employee count, salary,
department, hiring, attendance and workforce.

PROCUREMENT:
Purchase orders, procurement spend, vendors,
purchase approvals, purchasing and delivery.

FINANCE:
Invoices, payments, outstanding amounts,
accounts payable, expenses and financial amounts.

UNKNOWN:
Questions unrelated to these three domains.

Return ONLY one category name.
Do not explain your answer.

User question:
{question}
"""

    response = client.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You classify enterprise business questions."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={"temperature": 0}
    )

    domain = response.message.content.strip().upper()

    allowed_domains = [
        "HR",
        "PROCUREMENT",
        "FINANCE",
        "UNKNOWN"
    ]

    if domain not in allowed_domains:
        return "UNKNOWN"

    return domain
