
from app.sql_agent.domain_router import detect_domain

from app.agents.hr_sql_agent import run_hr_agent
from app.agents.procurement_sql_agent import run_procurement_agent
from app.agents.finance_sql_agent import run_finance_agent


AGENTS = {
    "HR": run_hr_agent,
    "PROCUREMENT": run_procurement_agent,
    "FINANCE": run_finance_agent
}


def main():

    print("========================================")
    print("     ENTERPRISE INTELLIGENT SQL AGENT")
    print("========================================")

    print("Supported domains: HR, Procurement, Finance")
    print("Type exit to stop.")

    while True:

        question = input("\nAsk your business question: ").strip()

        if question.lower() == "exit":
            print("Application stopped.")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:

            # Step 1: Identify business domain
            domain = detect_domain(question)

            print(f"\nDetected Domain: {domain}")

            if domain == "UNKNOWN":
                print(
                    "This question is outside the supported "
                    "HR, Procurement and Finance domains."
                )
                continue

            # Step 2: Route to specialist agent
            agent = AGENTS[domain]

            # Step 3: Execute specialist agent
            result = agent(question)

            print("\n========================================")
            print("BUSINESS ANSWER")
            print("========================================")

            print(result["answer"])

            print("\n========================================")
            print("GENERATED SQL")
            print("========================================")

            print(result["sql"])

            print("\n========================================")
            print("SNOWFLAKE QUERY RESULTS")
            print("========================================")

            print(result["results"])

        except Exception as error:

            print("\nAPPLICATION ERROR:")
            print(error)


if __name__ == "__main__":
    main()
