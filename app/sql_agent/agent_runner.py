
import time
import logging

from app.sql_agent.schema_loader import load_table_schemas
from app.sql_agent.sql_generator import generate_sql
from app.sql_agent.sql_validator import validate_sql
from app.sql_agent.snowflake_connection import execute_query
from app.sql_agent.result_explainer import explain_query_results

logger = logging.getLogger(__name__)


def run_domain_agent(question, domain, table):

    timings = {}

    # 1. Schema loading
    start = time.perf_counter()
    schemas = load_table_schemas()
    timings["schema_loading"] = round(time.perf_counter() - start, 3)

    if table not in schemas:
        raise ValueError(f"Table {table} is not configured.")

    domain_schema = {table: schemas[table]}

    # 2. SQL generation
    start = time.perf_counter()
    generated_sql = generate_sql(question, domain_schema)
    timings["sql_generation"] = round(time.perf_counter() - start, 3)

    # 3. SQL validation
    start = time.perf_counter()
    safe_sql = validate_sql(
        generated_sql,
        allowed_tables=[table]
    )
    timings["sql_validation"] = round(time.perf_counter() - start, 3)

    # 4. Snowflake execution
    start = time.perf_counter()
    results = execute_query(safe_sql)
    timings["snowflake_execution"] = round(time.perf_counter() - start, 3)

    # 5. Result explanation
    start = time.perf_counter()
    answer = explain_query_results(
        question,
        domain,
        safe_sql,
        results
    )
    timings["result_explanation"] = round(time.perf_counter() - start, 3)

    total = round(sum(timings.values()), 3)

    logger.info("SQL PERFORMANCE BREAKDOWN: %s", timings)
    logger.info("SQL TOTAL AGENT TIME: %s seconds", total)

    return {
        "domain": domain,
        "sql": safe_sql,
        "results": results,
        "answer": answer,
        "timings": timings,
        "total_agent_time": total
    }
