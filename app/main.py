
from ollama import Client
from config import OLLAMA_MODEL

# Connect to local Ollama server
client = Client(
    host="http://localhost:11434"
)


def ask_llm(question: str) -> str:
    response = client.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an Enterprise GenAI Assistant. "
                    "Provide accurate, structured answers. "
                    "Use practical enterprise examples."
                )
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )

    return response.message.content


def main():

    print("================================")
    print(" Enterprise GenAI Assistant")
    print(" Powered by Local Ollama")
    print(" Model:", OLLAMA_MODEL)
    print("================================")

    while True:

        question = input(
            "\nAsk your question (type exit to stop): "
        )

        if question.lower().strip() == "exit":
            print("Application stopped.")
            break

        if not question.strip():
            print("Please enter a question.")
            continue

        try:
            print("\nGenerating response...")

            answer = ask_llm(question)

            print("\nAI Response:")
            print(answer)

        except Exception as error:
            print("\nOllama Error:")
            print(error)


if __name__ == "__main__":
    main()
