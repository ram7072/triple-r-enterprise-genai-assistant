
import re
import sqlglot
from sqlglot import exp


def validate_sql(sql, allowed_tables):
    sql = re.sub(
        r"^```(?:sql)?\s*|\s*```$",
        "",
        sql.strip(),
        flags=re.IGNORECASE
    )

    statements = sqlglot.parse(sql, read="snowflake")
    if len(statements) != 1 or statements[0] is None:
        raise ValueError("Only one SQL statement is permitted.")
    query = statements[0]
    if not isinstance(query, exp.Select):
        raise ValueError("Only SELECT queries are permitted.")

    approved = {table.upper() for table in allowed_tables}

    referenced_tables = {
        table.name.upper()
        for table in query.find_all(exp.Table)
    }

    if not referenced_tables:
        raise ValueError("Query must reference an approved table.")

    unauthorized = referenced_tables - approved

    if unauthorized:
        raise ValueError(
            f"Access denied to tables: {unauthorized}"
        )

    if query.args.get("limit") is None:
        query = query.limit(100)

    return query.sql(dialect="snowflake")
