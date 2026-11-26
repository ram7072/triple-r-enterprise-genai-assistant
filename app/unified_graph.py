
import logging
import time
from pathlib import Path
from typing import TypedDict, Any

from ollama import Client
from langgraph.graph import StateGraph, START, END

from app.config import OLLAMA_MODEL
from app.intent_router import classify_intent
from app.rag import generate_rag_answer
from app.sql_agent.sql_graph import run_sql_graph
from app.monitoring import log_request


# ---------------------------------
# Logging Configuration
# ---------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=str(LOG_DIR / "assistant.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True
)

logger = logging.getLogger(__name__)

client = Client(host="http://localhost:11434")


# ---------------------------------
# Unified State
# ---------------------------------

class UnifiedState(TypedDict):
    question: str
    intent: str
    answer: str
    rag_answer: str
    sql_answer: str
    sql_query: str
    sources: list[dict[str, Any]]
    error: str


# ---------------------------------
# Supervisor
# ---------------------------------

def supervisor_node(state: UnifiedState):

    intent = classify_intent(state["question"])

    logger.info("Intent detected: %s", intent)

    return {"intent": intent}


# ---------------------------------
# RAG Agent
# ---------------------------------

def rag_node(state: UnifiedState):

    try:
        rag_result = generate_rag_answer(state["question"])

        answer = rag_result.get("answer", "")
        sources = rag_result.get("sources", [])

        logger.info(
            "RAG completed. Sources retrieved: %s",
            len(sources)
        )

        return {
            "rag_answer": answer,
            "answer": answer,
            "sources": sources,
            "error": ""
        }

    except Exception as error:

        logger.exception("RAG execution failed")

        return {
            "rag_answer": "",
            "answer": "Unable to retrieve the requested document information.",
            "sources": [],
            "error": f"RAG error: {error}"
        }


# ---------------------------------
# SQL Agent
# ---------------------------------

def sql_node(state: UnifiedState):

    try:
        sql_result = run_sql_graph(state["question"])

        answer = sql_result.get("answer", "")
        sql = sql_result.get("sql", "")

        logger.info(
            "SQL completed. Domain: %s",
            sql_result.get("domain", "UNKNOWN")
        )

        return {
            "sql_answer": answer,
            "sql_query": sql,
            "answer": answer,
            "sources": [],
            "error": ""
        }

    except Exception as error:

        logger.exception("SQL execution failed")

        return {
            "sql_answer": "",
            "sql_query": "",
            "answer": "Unable to retrieve the requested business data.",
            "sources": [],
            "error": f"SQL error: {error}"
        }


# ---------------------------------
# BOTH: RAG + SQL
# ---------------------------------

def both_node(state: UnifiedState):

    rag_answer = ""
    sql_answer = ""
    sql_query = ""
    sources = []
    errors = []

    # Run RAG
    try:
        rag_result = generate_rag_answer(state["question"])

        rag_answer = rag_result.get("answer", "")
        sources = rag_result.get("sources", [])

    except Exception as error:
        errors.append(f"RAG error: {error}")
        logger.exception("RAG failed in BOTH route")

    # Run SQL
    try:
        sql_result = run_sql_graph(state["question"])

        sql_answer = sql_result.get("answer", "")
        sql_query = sql_result.get("sql", "")

    except Exception as error:
        errors.append(f"SQL error: {error}")
        logger.exception("SQL failed in BOTH route")

    # Avoid synthesis if both agents failed
    if not rag_answer and not sql_answer:

        return {
            "rag_answer": "",
            "sql_answer": "",
            "sql_query": "",
            "sources": sources,
            "answer": "Both document retrieval and database retrieval failed.",
            "error": "; ".join(errors)
        }

    # Combine agent outputs
    prompt = f"""
You are an Enterprise AI Business Analyst.

Answer using only the supplied RAG and SQL outputs.

Rules:
1. Do not invent policies, numbers or business facts.
2. Distinguish document information from database results.
3. Mention relevant document source names and pages.
4. If one agent failed, disclose that limitation.
5. If information is unavailable, say so.

User Question:
{state["question"]}

Document Answer:
{rag_answer}

Snowflake Business Data Answer:
{sql_answer}

Provide one consolidated answer.
"""

    response = client.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        options={"temperature": 0.1}
    )

    logger.info(
        "BOTH route completed. Sources: %s",
        len(sources)
    )

    return {
        "rag_answer": rag_answer,
        "sql_answer": sql_answer,
        "sql_query": sql_query,
        "sources": sources,
        "answer": response.message.content,
        "error": "; ".join(errors)
    }


# ---------------------------------
# UNKNOWN
# ---------------------------------

def unknown_node(state: UnifiedState):

    return {
        "answer": (
            "I can assist with enterprise HR, Procurement "
            "and Finance questions using company documents "
            "and Snowflake business data."
        ),
        "sources": [],
        "error": ""
    }


# ---------------------------------
# Routing
# ---------------------------------

def route_intent(state: UnifiedState):
    return state["intent"]


# ---------------------------------
# Build Graph
# ---------------------------------

builder = StateGraph(UnifiedState)

builder.add_node("supervisor", supervisor_node)
builder.add_node("rag_agent", rag_node)
builder.add_node("sql_agent", sql_node)
builder.add_node("both_agents", both_node)
builder.add_node("unknown_agent", unknown_node)

builder.add_edge(START, "supervisor")

builder.add_conditional_edges(
    "supervisor",
    route_intent,
    {
        "RAG": "rag_agent",
        "SQL": "sql_agent",
        "BOTH": "both_agents",
        "UNKNOWN": "unknown_agent"
    }
)

builder.add_edge("rag_agent", END)
builder.add_edge("sql_agent", END)
builder.add_edge("both_agents", END)
builder.add_edge("unknown_agent", END)

unified_graph = builder.compile()


# ---------------------------------
# Public Entry Point
# ---------------------------------

def run_unified_assistant(question: str):

    start_time = time.perf_counter()

    initial_state = {
        "question": question,
        "intent": "",
        "answer": "",
        "rag_answer": "",
        "sql_answer": "",
        "sql_query": "",
        "sources": [],
        "error": ""
    }

    try:

        result = unified_graph.invoke(initial_state)

        elapsed = round(
            time.perf_counter() - start_time, 3
        )

        result["latency_seconds"] = elapsed

        error = result.get("error", "")

        status = "PARTIAL" if error else "SUCCESS"

        # Persist request metrics in SQLite
        try:
            log_request(
                question=question,
                intent=result.get("intent", "UNKNOWN"),
                status=status,
                latency_seconds=elapsed,
                sql_query=result.get("sql_query", ""),
                sources_count=len(result.get("sources", [])),
                error=error
            )

        except Exception:
            logger.exception("Unable to save monitoring record")

        logger.info(
            "Request completed | Intent: %s | Status: %s | Latency: %s",
            result.get("intent"),
            status,
            elapsed
        )

        return result

    except Exception as error:

        elapsed = round(
            time.perf_counter() - start_time, 3
        )

        logger.exception("Unified assistant request failed")

        # Record failed request
        try:
            log_request(
                question=question,
                intent="ERROR",
                status="FAILED",
                latency_seconds=elapsed,
                error=str(error)
            )

        except Exception:
            logger.exception("Unable to save failed monitoring record")

        raise
