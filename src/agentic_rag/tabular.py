"""Tabular ingestion: load CSV / Excel files into SQLite as queryable tables.

Text documents belong in the vector store; tabular data belongs in SQL,
where aggregations ("which product sold the most?") are exact instead of
an LLM guessing arithmetic over retrieved text chunks.
"""

import argparse
import logging
import re
import sqlite3
from pathlib import Path

from .config import settings

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".csv", ".xlsx", ".xls"}


def table_name_for(path: Path) -> str:
    """Derive a safe SQL table name from a file name."""
    name = re.sub(r"[^a-z0-9_]+", "_", path.stem.lower()).strip("_")
    if not name or name[0].isdigit():
        name = f"t_{name}"
    return name


def _load_dataframe(path: Path):
    import pandas as pd

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def ingest_tables(
    directory: str | None = None, sqlite_path: str | None = None
) -> dict[str, tuple[int, list[str]]]:
    """Load every CSV/Excel file in a directory tree into SQLite.

    Each file becomes one table (replaced if it already exists), named after
    the file. Returns {table_name: (row_count, column_names)}.
    """
    base = Path(directory or settings.tables_dir)
    results: dict[str, tuple[int, list[str]]] = {}
    if not base.exists():
        return results

    db_path = Path(sqlite_path or settings.sqlite_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    try:
        for path in sorted(base.rglob("*")):
            if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            frame = _load_dataframe(path)
            name = table_name_for(path)
            frame.to_sql(name, connection, if_exists="replace", index=False)
            results[name] = (len(frame), [str(c) for c in frame.columns])
            logger.info("Loaded %s -> table '%s' (%d rows)", path.name, name, len(frame))
    finally:
        connection.close()
    return results


def suggest_sql_description(results: dict[str, tuple[int, list[str]]]) -> str:
    """Draft a SQL_DESCRIPTION line so the router knows what the DB contains."""
    tables = "; ".join(
        f"{name} ({', '.join(columns)})" for name, (_, columns) in results.items()
    )
    return f"A SQLite database with the following tables: {tables}."


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Load CSV/Excel files into the agent's SQLite database."
    )
    parser.add_argument(
        "--dir", default=None, help=f"Directory to scan (default: {settings.tables_dir})"
    )
    parser.add_argument(
        "--db", default=None, help=f"SQLite file to write (default: {settings.sqlite_path})"
    )
    args = parser.parse_args()

    results = ingest_tables(args.dir, args.db)
    if not results:
        print(
            f"No CSV/Excel files found in {args.dir or settings.tables_dir}. "
            "Drop your .csv/.xlsx files there and run this command again."
        )
        return

    print(f"\n{len(results)} table(s) loaded into {args.db or settings.sqlite_path}:")
    for name, (rows, columns) in results.items():
        print(f"  {name}: {rows} rows ({', '.join(columns)})")

    print(
        "\nNow tell the router what the database contains. "
        "Set this in your .env (edit to taste):\n"
        f"SQL_DESCRIPTION={suggest_sql_description(results)}"
    )


if __name__ == "__main__":
    main()
