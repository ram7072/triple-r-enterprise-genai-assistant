
from app.sql_agent.agent_runner import run_domain_agent


def run_hr_agent(question):
    return run_domain_agent(
        question=question,
        domain="HR",
        table="EMPLOYEES"
    )
