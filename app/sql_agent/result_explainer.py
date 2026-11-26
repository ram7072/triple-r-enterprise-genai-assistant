
import json

from ollama import Client
from app.config import OLLAMA_MODEL

client = Client(host="http://localhost:11434")



def explain_query_results(question, domain, sql, results):

    if not results:
        return (
            "The Snowflake query executed successfully, "
            "but no records were found for your question."
        )

    row_count = len(results)

    # Single-row results: ideal for COUNT, SUM, AVG, etc.
    if row_count == 1:

        row = results[0]

        lines = [
            f"Business question: {question}",
            f"Domain: {domain}",
            "",
            "Query Result:"
        ]

        for column, value in row.items():
            lines.append(f"- {column}: {value}")

        return "\n".join(lines)

    # Multiple-row results
    columns = list(results[0].keys())

    lines = [
        f"Business question: {question}",
        f"Domain: {domain}",
        f"Records returned: {row_count}",
        "",
        "Results:"
    ]

    # Display up to 20 records
    display_rows = results[:20]

    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")

    for row in display_rows:

        values = [
            str(row.get(column, "")).replace("|", "\\|")
            for column in columns
        ]

        lines.append("| " + " | ".join(values) + " |")

    if row_count > 20:
        lines.append("")
        lines.append(
            f"Showing 20 of {row_count} returned records."
        )

    return "\n".join(lines)
