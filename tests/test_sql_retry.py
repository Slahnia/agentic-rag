"""SQL self-correction: failed queries are retried with error feedback."""

import sqlite3

import pytest
from langchain_community.utilities import SQLDatabase
from langchain_core.runnables import RunnableLambda

from agentic_rag.tools import sql as sql_module
from agentic_rag.tools.sql import SQLQuery, run_sql_question


class FakeStructuredLLM:
    """Returns scripted queries in order, recording the prompts it saw."""

    def __init__(self, queries):
        self.queries = list(queries)
        self.prompts = []

    def with_structured_output(self, schema):
        return RunnableLambda(self._respond)

    def _respond(self, prompt):
        self.prompts.append(prompt.to_string())
        return SQLQuery(query=self.queries.pop(0))


@pytest.fixture
def sample_db(tmp_path, monkeypatch):
    path = tmp_path / "test.db"
    connection = sqlite3.connect(path)
    connection.executescript(
        "CREATE TABLE items (name TEXT, qty INTEGER);"
        "INSERT INTO items VALUES ('a', 1), ('b', 2);"
    )
    connection.commit()
    connection.close()
    monkeypatch.setattr(
        sql_module, "get_db", lambda: SQLDatabase.from_uri(f"sqlite:///{path}")
    )


def test_retries_with_error_feedback_until_success(sample_db):
    llm = FakeStructuredLLM(
        ["SELECT nope FROM missing", "SELECT COUNT(*) FROM items"]
    )
    doc = run_sql_question("how many items?", llm)
    assert "2" in doc.page_content
    assert "Query failed" not in doc.page_content
    # The second prompt must contain the failed query and the error feedback
    assert "SELECT nope FROM missing" in llm.prompts[1]
    assert "previous query failed" in llm.prompts[1]


def test_gives_up_after_max_attempts(sample_db):
    llm = FakeStructuredLLM(["SELECT nope FROM missing"] * 3)
    doc = run_sql_question("how many items?", llm, max_attempts=3)
    assert "Query failed after 3 attempts" in doc.page_content


def test_empty_result_triggers_retry_with_join_hint(sample_db):
    llm = FakeStructuredLLM(
        ["SELECT name FROM items WHERE qty > 100", "SELECT name FROM items"]
    )
    doc = run_sql_question("list the items", llm)
    assert "a" in doc.page_content and "b" in doc.page_content
    assert "returned no rows" in llm.prompts[1]


def test_persistently_empty_result_is_reported_honestly(sample_db):
    llm = FakeStructuredLLM(["SELECT name FROM items WHERE qty > 100"] * 3)
    doc = run_sql_question("items with huge qty?", llm)
    assert "(no rows returned)" in doc.page_content


def test_unsafe_queries_are_never_executed(sample_db):
    llm = FakeStructuredLLM(["DROP TABLE items"] * 3)
    doc = run_sql_question("delete everything", llm)
    assert "read-only" in doc.page_content
