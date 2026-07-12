"""Graph nodes and routing decisions.

Nodes mutate state; decision functions are pure reads of the state so the
control flow can be unit-tested without an LLM.
"""

import logging

from langchain_core.documents import Document

from ..config import settings
from ..ingestion import get_retriever
from ..tools.sql import run_sql_question
from ..tools.web_search import search_web
from .chains import (
    get_document_grader,
    get_generator,
    get_grounding_grader,
    get_rewriter,
    get_router,
    get_sql_llm,
)
from .state import GraphState

logger = logging.getLogger(__name__)


def _format_docs(documents: list[Document]) -> str:
    return "\n\n".join(
        f"[{doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in documents
    )


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------
def route(state: GraphState) -> GraphState:
    """Decide which data source should handle the question."""
    result = get_router().invoke({"question": state["question"]})
    logger.info("Router -> %s", result.datasource)
    return {
        "datasource": result.datasource,
        "original_question": state["question"],
        "retries": 0,
    }


def retrieve(state: GraphState) -> GraphState:
    documents = get_retriever().invoke(state["question"])
    logger.info("Retrieved %d chunks", len(documents))
    return {"documents": documents}


def grade_documents(state: GraphState) -> GraphState:
    """Keep only the retrieved chunks an LLM grader judges relevant."""
    grader = get_document_grader()
    relevant = [
        doc
        for doc in state["documents"]
        if grader.invoke({"document": doc.page_content, "question": state["question"]})
        == "yes"
    ]
    logger.info("Document grading: %d/%d relevant", len(relevant), len(state["documents"]))
    return {"documents": relevant}


def rewrite_query(state: GraphState) -> GraphState:
    rewritten = get_rewriter().invoke(
        {
            "original_question": state["original_question"],
            "question": state["question"],
        }
    ).strip()
    logger.info("Rewrote query -> %s", rewritten)
    return {"question": rewritten, "retries": state.get("retries", 0) + 1}


def web_search(state: GraphState) -> GraphState:
    documents = search_web(state["question"])
    logger.info("Web search returned %d results", len(documents))
    # Mark the source so downstream decisions know we are on the web path
    # (this node also serves as fallback when vectorstore retrieval fails).
    return {"documents": documents, "datasource": "web_search"}


def query_sql(state: GraphState) -> GraphState:
    document = run_sql_question(state["question"], get_sql_llm())
    return {"documents": [document]}


def generate(state: GraphState) -> GraphState:
    generation = get_generator().invoke(
        {
            "context": _format_docs(state["documents"]),
            "question": state.get("original_question", state["question"]),
        }
    )
    return {"generation": generation}


# --------------------------------------------------------------------------
# Routing decisions (pure functions over the state)
# --------------------------------------------------------------------------
def select_datasource(state: GraphState) -> str:
    return state["datasource"]


def decide_after_grading(state: GraphState) -> str:
    """After grading: answer, retry with a better query, or fall back to web."""
    if state["documents"]:
        return "generate"
    if state.get("retries", 0) < settings.max_retries:
        return "rewrite"
    logger.info("No relevant documents after %d retries, falling back to web", settings.max_retries)
    return "web_search"


def route_after_rewrite(state: GraphState) -> str:
    """Send the rewritten query back to the source that produced bad results."""
    return "web_search" if state["datasource"] == "web_search" else "retrieve"


def grade_generation(state: GraphState) -> str:
    """Check the answer is grounded in the evidence; retry once if not.

    SQL results are deterministic query output, so grounding them against an
    LLM grader adds latency without value — accept them directly.
    """
    if state["datasource"] == "sql" or not state["documents"]:
        return "useful"
    grounded = get_grounding_grader().invoke(
        {
            "documents": _format_docs(state["documents"]),
            "generation": state["generation"],
        }
    )
    if grounded == "yes":
        return "useful"
    if state.get("retries", 0) < settings.max_retries:
        logger.info("Answer not grounded, rewriting query and retrying")
        return "rewrite"
    logger.info("Answer not grounded but retry budget exhausted, returning anyway")
    return "useful"
