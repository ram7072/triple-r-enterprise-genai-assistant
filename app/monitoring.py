
import sqlite3
from pathlib import Path
from datetime import datetime, timezone


# ---------------------------------
# Database Configuration
# ---------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

DB_PATH = LOG_DIR / "monitoring.db"


# ---------------------------------
# Database Connection
# ---------------------------------

def get_connection():

    connection = sqlite3.connect(
        str(DB_PATH),
        timeout=10
    )

    connection.row_factory = sqlite3.Row

    return connection


# ---------------------------------
# Initialize Monitoring Database
# ---------------------------------

def init_db():

    with get_connection() as connection:

        connection.execute("""
            CREATE TABLE IF NOT EXISTS assistant_requests (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                timestamp TEXT NOT NULL,

                question TEXT,

                intent TEXT,

                status TEXT,

                latency_seconds REAL,

                sql_query TEXT,

                sources_count INTEGER DEFAULT 0,

                error TEXT
            )
        """)

        connection.commit()


# ---------------------------------
# Record Request
# ---------------------------------

def log_request(
    question,
    intent,
    status,
    latency_seconds,
    sql_query="",
    sources_count=0,
    error=""
):

    init_db()

    timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    with get_connection() as connection:

        connection.execute("""
            INSERT INTO assistant_requests (
                timestamp,
                question,
                intent,
                status,
                latency_seconds,
                sql_query,
                sources_count,
                error
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp,
            question,
            intent,
            status,
            latency_seconds,
            sql_query,
            sources_count,
            error
        ))

        connection.commit()


# ---------------------------------
# Dashboard KPIs
# ---------------------------------

def get_dashboard_metrics():

    init_db()

    with get_connection() as connection:

        row = connection.execute("""
            SELECT
                COUNT(*) AS total_requests,

                SUM(
                    CASE WHEN status = 'SUCCESS'
                    THEN 1 ELSE 0 END
                ) AS successful_requests,

                SUM(
                    CASE WHEN status = 'FAILED'
                    THEN 1 ELSE 0 END
                ) AS failed_requests,

                AVG(latency_seconds) AS average_latency

            FROM assistant_requests
        """).fetchone()

    return {
        "total_requests": row["total_requests"] or 0,
        "successful_requests": row["successful_requests"] or 0,
        "failed_requests": row["failed_requests"] or 0,
        "average_latency": round(
            row["average_latency"] or 0, 2
        )
    }


# ---------------------------------
# Intent Distribution
# ---------------------------------

def get_intent_distribution():

    init_db()

    with get_connection() as connection:

        rows = connection.execute("""
            SELECT
                intent,
                COUNT(*) AS request_count

            FROM assistant_requests

            GROUP BY intent
            ORDER BY request_count DESC
        """).fetchall()

    return [
        dict(row) for row in rows
    ]


# ---------------------------------
# Recent Request History
# ---------------------------------

def get_recent_requests(limit=50):

    init_db()

    with get_connection() as connection:

        rows = connection.execute("""
            SELECT *
            FROM assistant_requests
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()

    return [
        dict(row) for row in rows
    ]


# ---------------------------------
# Daily Request Trend
# ---------------------------------

def get_daily_trend():

    init_db()

    with get_connection() as connection:

        rows = connection.execute("""
            SELECT
                DATE(timestamp) AS request_date,
                COUNT(*) AS total_requests,
                AVG(latency_seconds) AS average_latency

            FROM assistant_requests

            GROUP BY DATE(timestamp)

            ORDER BY request_date
        """).fetchall()

    return [
        dict(row) for row in rows
    ]


# ---------------------------------
# Test Execution
# ---------------------------------

if __name__ == "__main__":

    init_db()

    print("AI CONTROL TOWER DATABASE")
    print("-------------------------")
    print("Database:", DB_PATH)

    print("\nCurrent Dashboard Metrics:")
    print(get_dashboard_metrics())

    print("\nRecent Requests:")
    print(get_recent_requests())
