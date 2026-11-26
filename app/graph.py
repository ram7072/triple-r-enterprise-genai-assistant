
from typing import TypedDict, Literal

from ollama import Client
from langgraph.graph import StateGraph, START, END

from app.config import OLLAMA_MODEL
from app.policy_agents import (
    hr_agent,
    procurement_agent,
    finance_agent,
    general_agent
)

client = Client(host="http://localhost:11434")


class AssistantState(TypedDict):
    question: str
    department: str
    answer: str
    sources: list


def supervisor(state: AssistantState):

    question = state["question"]

    response = client.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "system",
                "content": """
You are an Enterprise AI Supervisor.

Classify the user's question into exactly one category:

HR
PROCUREMENT
FINANCE
GENERAL

HR: employee, leave, attendance, working hours.
PROCUREMENT: vendors, purchase orders, approvals.
FINANCE: invoices, payments, expenses, accounts payable.
GENERAL: anything else.

Return ONLY the category name.
"""
            },
            {
                "role": "user",
                "content": question
            }
        ],
        options={"temperature": 0}
    )

    department = response.message.content.strip().lower()

    allowed = ["hr", "procurement", "finance", "general"]

    if department not in allowed:
        department = "general"

    print(f"\nSupervisor selected: {department.upper()} AGENT")

    return {"department": department}


def execute_hr(state: AssistantState):
    result = hr_agent(state["question"])
    return {
        "answer": result["answer"],
        "sources": result["sources"]
    }


def execute_procurement(state: AssistantState):
    result = procurement_agent(state["question"])
    return {
        "answer": result["answer"],
        "sources": result["sources"]
    }


def execute_finance(state: AssistantState):
    result = finance_agent(state["question"])
    return {
        "answer": result["answer"],
        "sources": result["sources"]
    }


def execute_general(state: AssistantState):
    result = general_agent(state["question"])
    return {
        "answer": result["answer"],
        "sources": result["sources"]
    }


builder = StateGraph(AssistantState)

builder.add_node("supervisor", supervisor)
builder.add_node("hr_agent", execute_hr)
builder.add_node("procurement_agent", execute_procurement)
builder.add_node("finance_agent", execute_finance)
builder.add_node("general_agent", execute_general)

builder.add_edge(START, "supervisor")

builder.add_conditional_edges(
    "supervisor",
    lambda state: state["department"],
    {
        "hr": "hr_agent",
        "procurement": "procurement_agent",
        "finance": "finance_agent",
        "general": "general_agent"
    }
)

builder.add_edge("hr_agent", END)
builder.add_edge("procurement_agent", END)
builder.add_edge("finance_agent", END)
builder.add_edge("general_agent", END)

graph = builder.compile()


if __name__ == "__main__":

    print("====================================")
    print(" LANGGRAPH ENTERPRISE AI ASSISTANT")
    print("====================================")

    while True:

        question = input("\nAsk a question (exit to stop): ")

        if question.lower().strip() == "exit":
            break

        if not question.strip():
            continue

        try:
            result = graph.invoke({
                "question": question,
                "department": "",
                "answer": "",
                "sources": []
            })

            print("\nSelected Agent:", result["department"].upper())
            print("\nAI Answer:")
            print(result["answer"])

            print("\nSources:")
            for source in result["sources"]:
                print(
                    source["document"],
                    "| Page:", source["page"]
                )

        except Exception as error:
            print("Error:", error)
