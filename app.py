"""Streamlit chat UI for the agentic RAG graph.

Run with: streamlit run app.py
"""

import streamlit as st

from agentic_rag.graph.build import build_graph
from agentic_rag.ingestion import ensure_index

NODE_LABELS = {
    "route": "🧭 Routing the question…",
    "retrieve": "📚 Searching the knowledge base…",
    "grade_documents": "🔍 Grading retrieved documents…",
    "rewrite_query": "✏️ Rewriting the query…",
    "web_search": "🌐 Searching the web…",
    "query_sql": "🗄️ Querying the database…",
    "generate": "💬 Writing the answer…",
}

st.set_page_config(page_title="Agentic RAG", page_icon="🧭", layout="centered")
st.title("🧭 Multi-source Agentic RAG")
st.caption(
    "LangGraph agent that routes each question to a vector store, the web or "
    "a SQL database — with retrieval grading and self-correction. "
    "Runs fully local on CPU."
)


@st.cache_resource(show_spinner="Loading models and index…")
def get_graph():
    ensure_index()
    return build_graph()


graph = get_graph()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Ask me anything…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        state: dict = {}
        with st.status("Thinking…", expanded=True) as status:
            for update in graph.stream({"question": question}, stream_mode="updates"):
                for node, values in update.items():
                    st.write(NODE_LABELS.get(node, node))
                    if values:
                        state.update(values)
            status.update(label="Done", state="complete", expanded=False)

        answer = state.get("generation", "Something went wrong — no answer produced.")
        st.markdown(answer)

        documents = state.get("documents", [])
        if documents:
            with st.expander(f"Evidence ({state.get('datasource', '?')} — {len(documents)} items)"):
                for doc in documents:
                    st.markdown(f"**{doc.metadata.get('source', 'unknown')}**")
                    st.text(doc.page_content[:500])
                    st.divider()

    st.session_state.messages.append({"role": "assistant", "content": answer})
