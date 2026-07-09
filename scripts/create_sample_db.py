"""Create the sample SQLite database used by the SQL data source."""

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agentic_rag.config import settings  # noqa: E402

PRODUCTS = [
    (1, "Mechanical Keyboard TKL", "peripherals", 89.90),
    (2, "Wireless Mouse Pro", "peripherals", 49.50),
    (3, "27-inch 4K Monitor", "displays", 329.00),
    (4, "USB-C Docking Station", "accessories", 119.00),
    (5, "Noise-Cancelling Headset", "audio", 149.99),
    (6, "Webcam 1080p", "peripherals", 59.00),
    (7, "Standing Desk Frame", "furniture", 399.00),
    (8, "Ergonomic Chair", "furniture", 289.00),
    (9, "Portable SSD 1TB", "storage", 99.00),
    (10, "Laptop Stand Aluminium", "accessories", 39.90),
]

SALES = [
    # (id, product_id, quantity, sale_date, country)
    (1, 1, 3, "2026-01-12", "Spain"),
    (2, 3, 1, "2026-01-15", "Mexico"),
    (3, 5, 2, "2026-01-20", "Spain"),
    (4, 2, 5, "2026-02-02", "Argentina"),
    (5, 9, 4, "2026-02-10", "Spain"),
    (6, 7, 1, "2026-02-14", "Chile"),
    (7, 8, 2, "2026-02-21", "Mexico"),
    (8, 1, 2, "2026-03-01", "Colombia"),
    (9, 4, 3, "2026-03-05", "Spain"),
    (10, 6, 6, "2026-03-09", "Mexico"),
    (11, 3, 2, "2026-03-18", "Spain"),
    (12, 10, 8, "2026-04-02", "Argentina"),
    (13, 5, 1, "2026-04-11", "Chile"),
    (14, 9, 2, "2026-04-19", "Mexico"),
    (15, 2, 3, "2026-05-06", "Spain"),
    (16, 7, 1, "2026-05-15", "Colombia"),
    (17, 1, 4, "2026-05-23", "Mexico"),
    (18, 8, 1, "2026-06-04", "Spain"),
    (19, 3, 3, "2026-06-17", "Argentina"),
    (20, 5, 2, "2026-06-28", "Spain"),
]


def main() -> None:
    path = Path(settings.sqlite_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    cursor = connection.cursor()
    cursor.executescript(
        """
        DROP TABLE IF EXISTS sales;
        DROP TABLE IF EXISTS products;
        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            price REAL NOT NULL
        );
        CREATE TABLE sales (
            id INTEGER PRIMARY KEY,
            product_id INTEGER NOT NULL REFERENCES products(id),
            quantity INTEGER NOT NULL,
            sale_date TEXT NOT NULL,
            country TEXT NOT NULL
        );
        """
    )
    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?)", PRODUCTS)
    cursor.executemany("INSERT INTO sales VALUES (?, ?, ?, ?, ?)", SALES)
    connection.commit()
    connection.close()
    print(f"Sample database created at {path}")


if __name__ == "__main__":
    main()
