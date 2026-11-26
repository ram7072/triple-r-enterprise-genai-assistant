
from ollama import Client
from app.config import OLLAMA_MODEL

client = Client(host="http://localhost:11434")


def generate_sql(question, schemas):

    prompt = f"""
You are an enterprise Snowflake SQL expert.

Available database schema:

{schemas}

Generate a Snowflake SQL query for the following question.

Rules:

1. Use only the supplied tables.
2. Use only the supplied columns.
3. Generate only one SELECT statement.
4. Never generate INSERT, UPDATE, DELETE,
   DROP, ALTER, CREATE or MERGE.
5. Use Snowflake SQL syntax.
6. Do not invent column names.
7. Use fully qualified table names where appropriate.
8. Limit output to 100 rows.
9. Return only SQL without Markdown.

User question:
{question}
"""

    response = client.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Generate accurate, read-only Snowflake SQL."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={"temperature": 0}
    )

    return response.message.content.strip()
