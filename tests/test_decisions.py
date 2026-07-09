"""Unit tests for the graph's routing decisions (no LLM required)."""

from langchain_core.documents import Document

from agentic_rag.config import settings
from agentic_rag.graph.nodes import (
    decide_after_grading,
    route_after_rewrite,
    select_datasource,
)


def test_select_datasource_reads_state():
    assert select_datasource({"datasource": "sql"}) == "sql"


def test_generate_when_relevant_documents_survive():
    state = {"documents": [Document(page_content="relevant")], "retries": 0}
    assert decide_after_grading(state) == "generate"


def test_rewrite_when_no_documents_and_budget_left():
    state = {"documents": [], "retries": 0}
    assert decide_after_grading(state) == "rewrite"


def test_web_fallback_when_retry_budget_exhausted():
    state = {"documents": [], "retries": settings.max_retries}
    assert decide_after_grading(state) == "web_search"


def test_rewrite_goes_back_to_original_source():
    assert route_after_rewrite({"datasource": "vectorstore"}) == "retrieve"
    assert route_after_rewrite({"datasource": "web_search"}) == "web_search"
