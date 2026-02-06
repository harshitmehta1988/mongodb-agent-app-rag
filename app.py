"""Streamlit UI for the MongoDB natural-language query agent (with RAG)."""
import streamlit as st
from agent import run_agent
from langchain_core.messages import ToolMessage, AIMessage

st.set_page_config(page_title="Inferyx Query Agent", page_icon="🔮", layout="centered")

st.title("🔮 Inferyx MongoDB Query Agent")
st.caption("Ask in plain English. The agent will discover schema, build find/aggregation (including joins), and return results.")

if "history" not in st.session_state:
    st.session_state.history = []

prompt = st.text_area(
    "Your question",
    placeholder="e.g. List all users. / How many orders per customer? / Join orders with users and show name and total.",
    height=120,
)
col1, col2, _ = st.columns([1, 1, 3])
run = col1.button("Run", type="primary")
show_trace = col2.checkbox("Show tool trace", value=False)

if run and prompt.strip():
    with st.spinner("Thinking and querying..."):
        try:
            final_text, messages = run_agent(prompt.strip())
            st.session_state.last_result = (final_text, messages)
        except Exception as e:
            st.session_state.last_result = (None, None)
            st.error(str(e))
            st.stop()

    final_text, messages = st.session_state.last_result
    if final_text:
        st.subheader("Answer")
        st.markdown(final_text)

    if show_trace and messages:
        with st.expander("Tool trace (queries & tool results)"):
            for m in messages:
                if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                    for tc in m.tool_calls:
                        st.code(
                            f"Tool: {tc.get('name')}\nArgs: {tc.get('args')}",
                            language="json",
                        )
                if isinstance(m, ToolMessage):
                    content = m.content
                    if len(content) > 2000:
                        content = content[:2000] + "\n... (truncated)"
                    st.text(content)
                    st.divider()

elif run and not prompt.strip():
    st.warning("Please enter a question.")

st.divider()
st.markdown("**Examples:** *List collections*, *What collections exist?*, *Show schema of X*, *Find documents in X where ...*, *Join X and Y on field Z*, *Count by category*")
