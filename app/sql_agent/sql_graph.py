
from typing import TypedDict, Any
from langgraph.graph import StateGraph, START, END
from app.sql_agent.domain_router import detect_domain
from app.agents.hr_sql_agent import run_hr_agent
from app.agents.procurement_sql_agent import run_procurement_agent
from app.agents.finance_sql_agent import run_finance_agent


# -----------------------------------------
# 1. Define shared graph state
# -----------------------------------------

class SQLAgentState(TypedDict):
    question: str
    domain: str
    answer: str
    sql: str
    results: list[Any]
    error: str


# -----------------------------------------
# 2. Supervisor node
# -----------------------------------------

def supervisor_node(state: SQLAgentState):

    domain = detect_domain(state["question"])

    print(f"\nSupervisor selected: {domain}")

    return {
        "domain": domain
    }


# -----------------------------------------
# 3. HR specialist node
# -----------------------------------------

def hr_node(state: SQLAgentState):

    result = run_hr_agent(state["question"])

    return {
        "answer": result["answer"],
        "sql": result["sql"],
        "results": result["results"],
        "error": ""
    }


# -----------------------------------------
# 4. Procurement specialist node
# -----------------------------------------

def procurement_node(state: SQLAgentState):

    result = run_procurement_agent(state["question"])

    return {
        "answer": result["answer"],
        "sql": result["sql"],
        "results": result["results"],
        "error": ""
    }


# -----------------------------------------
# 5. Finance specialist node
# -----------------------------------------

def finance_node(state: SQLAgentState):

    result = run_finance_agent(state["question"])

    return {
        "answer": result["answer"],
        "sql": result["sql"],
        "results": result["results"],
        "error": ""
    }


# -----------------------------------------
# 6. Unknown domain node
# -----------------------------------------

def unknown_node(state: SQLAgentState):

    return {
        "answer": (
            "This question is outside the supported "
            "HR, Procurement and Finance domains."
        ),
        "sql": "",
        "results": [],
        "error": ""
    }


# -----------------------------------------
# 7. Build LangGraph workflow
# -----------------------------------------

builder = StateGraph(SQLAgentState)

builder.add_node("supervisor", supervisor_node)

builder.add_node("hr_agent", hr_node)

builder.add_node("procurement_agent", procurement_node)

builder.add_node("finance_agent", finance_node)

builder.add_node("unknown_agent", unknown_node)


# Entry point
builder.add_edge(START, "supervisor")


# Conditional routing
def route_question(state: SQLAgentState):

    return state["domain"]


builder.add_conditional_edges(
    "supervisor",
    route_question,
    {
        "HR": "hr_agent",
        "PROCUREMENT": "procurement_agent",
        "FINANCE": "finance_agent",
        "UNKNOWN": "unknown_agent"
    }
)


# Connect specialist agents to END
builder.add_edge("hr_agent", END)

builder.add_edge("procurement_agent", END)

builder.add_edge("finance_agent", END)

builder.add_edge("unknown_agent", END)


# Compile graph
sql_graph = builder.compile()


# -----------------------------------------
# 8. Execute graph
# -----------------------------------------

def run_sql_graph(question: str):

    initial_state = {
        "question": question,
        "domain": "",
        "answer": "",
        "sql": "",
        "results": [],
        "error": ""
    }

    return sql_graph.invoke(initial_state)


# -----------------------------------------
# 9. Interactive application
# -----------------------------------------

if __name__ == "__main__":

    print("========================================")
    print(" LANGGRAPH ENTERPRISE SQL SUPERVISOR")
    print(" HR | PROCUREMENT | FINANCE")
    print("========================================")

    while True:

        question = input("\nAsk your business question: ").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        try:

            response = run_sql_graph(question)

            print("\nDOMAIN:", response["domain"])

            print("\nBUSINESS ANSWER:")
            print(response["answer"])

            if response["sql"]:

                print("\nGENERATED SQL:")
                print(response["sql"])

                print("\nSNOWFLAKE RESULTS:")
                print(response["results"])

        except Exception as error:

            print("\nAPPLICATION ERROR:")
            print(error)
