
import os
import re
from dotenv import load_dotenv
if __package__:
    from .snowflake_connection import get_connection
else:
    from snowflake_connection import get_connection

load_dotenv()


def get_allowed_tables():

    return [
        table.strip().upper()
        for table in os.getenv(
            "SNOWFLAKE_ALLOWED_TABLES", ""
        ).split(",")
        if table.strip()
    ]


def load_table_schemas():
    tables = get_allowed_tables()
    if not tables:
        raise ValueError("No approved tables configured.")
    schemas = {}
    connection = get_connection()
    try:
        cursor = connection.cursor()
        for table in tables:
            if not re.fullmatch(r"[A-Z0-9_]+", table):
                raise ValueError("Invalid table identifier.")
            cursor.execute(f"DESCRIBE TABLE {table}")
            columns = cursor.fetchall()
            schemas[table] = [
                {
                    "column_name": row[0],
                    "data_type": row[1]
                }
                for row in columns
            ]

        cursor.close()
    finally:
        connection.close()
    ##print(f"Loaded schemas for tables: {', '.join(schemas.keys())}")
    return schemas
load_table_schemas()