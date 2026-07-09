"""Shared state that flows through the LangGraph nodes."""

from typing import TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict, total=False):
    question: str  # current (possibly rewritten) question used for retrieval
    original_question: str  # what the user actually asked
    datasource: str  # "vectorstore" | "web_search" | "sql"
    documents: list[Document]  # evidence gathered for the answer
    generation: str  # final answer
    retries: int  # rewrite attempts so far (bounded by settings.max_retries)
