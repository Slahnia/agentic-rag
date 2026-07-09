"""CSV/Excel ingestion into SQLite."""

import sqlite3
from pathlib import Path

from agentic_rag.tabular import ingest_tables, suggest_sql_description, table_name_for


def test_table_name_sanitization():
    assert table_name_for(Path("My Sales-2026.csv")) == "my_sales_2026"
    assert table_name_for(Path("2026 sales!.csv")) == "t_2026_sales"
    assert table_name_for(Path("ventas.xlsx")) == "ventas"


def test_ingest_csv_creates_queryable_table(tmp_path):
    (tmp_path / "tables").mkdir()
    (tmp_path / "tables" / "Monthly Revenue.csv").write_text(
        "month,revenue\n2026-01,1000\n2026-02,1500\n", encoding="utf-8"
    )
    db = tmp_path / "test.db"

    results = ingest_tables(str(tmp_path / "tables"), str(db))

    assert results == {"monthly_revenue": (2, ["month", "revenue"])}
    connection = sqlite3.connect(db)
    total = connection.execute("SELECT SUM(revenue) FROM monthly_revenue").fetchone()[0]
    connection.close()
    assert total == 2500


def test_ingest_missing_directory_returns_empty(tmp_path):
    assert ingest_tables(str(tmp_path / "nope"), str(tmp_path / "test.db")) == {}


def test_suggested_description_mentions_tables_and_columns():
    text = suggest_sql_description({"ventas": (10, ["producto", "importe"])})
    assert "ventas" in text and "producto" in text and "importe" in text
