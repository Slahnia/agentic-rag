"""The SQL source must only ever execute read-only queries."""

import pytest

from agentic_rag.tools.sql import is_safe_select


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM products",
        "select name, price from products where price > 100;",
        "WITH top AS (SELECT * FROM sales) SELECT * FROM top",
        "  SELECT count(*) FROM sales GROUP BY country  ",
    ],
)
def test_accepts_read_only_queries(sql):
    assert is_safe_select(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE products",
        "DELETE FROM sales",
        "INSERT INTO products VALUES (99, 'x', 'y', 1)",
        "UPDATE products SET price = 0",
        "SELECT * FROM products; DROP TABLE products",
        "PRAGMA writable_schema = ON",
        "ATTACH DATABASE 'evil.db' AS evil",
    ],
)
def test_rejects_write_and_multi_statement_queries(sql):
    assert not is_safe_select(sql)
