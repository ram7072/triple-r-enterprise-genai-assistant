
import streamlit as st

from app.unified_graph import run_unified_assistant
from app.context_resolver import resolve_context

# ---------------------------------
# Page Configuration
# ---------------------------------

st.set_page_config(
    page_title="Triple-R Enterprise GenAI Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("Triple-R Enterprise GenAI Assistant")

st.caption(
    "Unified AI Assistant | HR | Procurement | Finance | "
    "RAG + Snowflake SQL | Local Ollama"
)


# ---------------------------------
# Sidebar
# ---------------------------------

with st.sidebar:

    st.header("Assistant Configuration")

    st.success("Local LLM: Ollama")
    st.info("Model: Llama 3.2:3B")
    st.info("Vector DB: FAISS")
    st.info("Embedding: Nomic")
    st.info("Database: Snowflake")

    st.divider()

    st.subheader("Available Agents")

    st.write("📚 RAG Agent")
    st.write("🗄️ SQL Agent")
    st.write("🔀 Unified Supervisor")

    st.divider()

    if st.button("Clear conversation", use_container_width=True):

        st.session_state.messages = []
        st.rerun()


# ---------------------------------
# Initialize Session State
# ---------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------------
# Display Previous Conversations
# ---------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("intent"):
            st.caption(f"Detected Intent: {message['intent']}")

        if message.get("sql"):
            with st.expander("View Generated SQL"):
                st.code(message["sql"], language="sql")
        if message.get("sources"):
            with st.expander("View document citations"):
                for source in message["sources"]:
                    st.write(
                        f"Document: {source['document']} | "
                        f"Page: {source['page']} | "
                        f"Similarity: {source['score']}"
                    )

        if message.get("latency") is not None:
            st.caption(f"Response time: {message['latency']} seconds")

        if message.get("error"):
            with st.expander("View Warnings"):
                st.warning(message["error"])


# ---------------------------------
# User Input
# ---------------------------------

question = st.chat_input(
    "Ask about HR, Procurement, Finance or company policies..."
)


# ---------------------------------
# Process User Question
# ---------------------------------

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner(
            "Supervisor analyzing your question and routing to agents..."
        ):

            try:

                # Invoke Unified LangGraph Supervisor
                history = st.session_state.messages[:-1]
                resolved_question = resolve_context(
                    question,
                    history
                )
                st.caption(f"Interpreted question: {resolved_question}")
                result = run_unified_assistant(resolved_question)

                answer = result.get(
                    "answer",
                    "No answer was generated."
                )

                intent = result.get("intent", "UNKNOWN")
                sql = result.get("sql_query", "")
                error = result.get("error", "")
                sources = result.get("sources", [])
                latency = result.get("latency_seconds")

                # Display Business Answer
                st.markdown(answer)

                # Display Intent
                st.caption(f"Handled by: {intent} route")

                # Display Generated SQL
                if sql:

                    with st.expander("View Generated SQL"):
                        st.code(sql, language="sql")
                if sources:
                    with st.expander("View document citations"):
                        for source in sources:
                            st.write(
                                f"Document: {source['document']} | "
                                f"Page: {source['page']} | "
                                f"Similarity: {source['score']}"
                            )

                if latency is not None:
                    st.caption(f"Response time: {latency} seconds")

                # Display Errors / Warnings
                if error:

                    with st.expander("View Warnings"):
                        st.warning(error)

                # Save Assistant Response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "intent": intent,
                    "sql": sql,
                    "error": error
                })

            except Exception as error:

                error_message = str(error)

                st.error(
                    f"Application error: {error_message}"
                )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Application error: {error_message}"
                })
