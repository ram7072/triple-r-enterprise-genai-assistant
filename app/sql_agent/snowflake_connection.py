
import os
import snowflake.connector
from dotenv import load_dotenv

load_dotenv()


def get_connection():
    return snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
        role=os.getenv("SNOWFLAKE_ROLE")
    )


def execute_query(sql):
    connection = get_connection()
    try:
        cursor = connection.cursor()
        cursor.execute(sql)
        columns = [
            column[0] for column in cursor.description
        ]
        rows = cursor.fetchmany(100)
        results = [
            dict(zip(columns, row))
            for row in rows
        ]
        cursor.close()
        return results
    finally:
        connection.close()
