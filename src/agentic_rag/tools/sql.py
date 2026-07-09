"""Text-to-SQL source over a local SQLite database (read-only by design)."""

import logging
import re
from functools import lru_cache

from langchain_community.utilities import SQLDatabase
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from ..config import settings

logger = logging.getLogger(__name__)

FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|pragma|vacuum)\b",
    re.IGNORECASE,
)

SQL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert SQLite analyst. Given the database schema below, "
            "write ONE syntactically correct SELECT query that answers the user "
            "question. Only use tables and columns that exist in the schema. "
            "Limit results to at most 20 rows unless the question asks otherwise."
            "\n\nSchema:\n{schema}",
        ),
        ("human", "{question}"),
    ]
)


class SQLQuery(BaseModel):
    """A single read-only SQL query."""

    query: str = Field(description="The SELECT statement to execute.")


def is_safe_select(sql: str) -> bool:
    """Accept only single SELECT statements with no write/DDL keywords."""
    statement = sql.strip().rstrip(";")
    if ";" in statement:  # multiple statements
        return False
    if not statement.lower().lstrip("( ").startswith(("select", "with")):
        return False
    return not FORBIDDEN.search(statement)


@lru_cache(maxsize=1)
def get_db() -> SQLDatabase:
    return SQLDatabase.from_uri(f"sqlite:///{settings.sqlite_path}")


def run_sql_question(question: str, llm) -> Document:
    """Generate a SQL query for the question, execute it and wrap the result."""
    db = get_db()
    chain = SQL_PROMPT | llm.with_structured_output(SQLQuery)
    sql = chain.invoke({"schema": db.get_table_info(), "question": question}).query
    logger.info("Generated SQL: %s", sql)

    if not is_safe_select(sql):
        return Document(
            page_content="The generated SQL query was rejected because it was "
            "not a single read-only SELECT statement.",
            metadata={"source": "sql", "query": sql},
        )

    try:
        result = db.run(sql)
    except Exception as exc:
        result = f"Query failed: {exc}"
    return Document(
        page_content=f"SQL query:\n{sql}\n\nResult:\n{result}",
        metadata={"source": "sql", "query": sql},
    )
