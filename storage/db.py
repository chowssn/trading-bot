"""SQLite persistence for quotes, signals, and orders. No ORM."""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from config.settings import DB_PATH
from execution.broker_adapter import OrderResult, Quote

SCHEMA = """
CREATE TABLE IF NOT EXISTS quotes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    bid REAL NOT NULL,
    ask REAL NOT NULL,
    last REAL NOT NULL,
    timestamp TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_quotes_symbol_timestamp ON quotes (symbol, timestamp);

CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    signal_name TEXT NOT NULL,
    value REAL NOT NULL,
    timestamp TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_signals_symbol_timestamp ON signals (symbol, timestamp);

CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity REAL NOT NULL,
    status TEXT NOT NULL,
    filled_qty REAL,
    avg_fill_price REAL,
    paper_mode INTEGER NOT NULL,
    timestamp TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_orders_symbol_timestamp ON orders (symbol, timestamp);
"""


class Database:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def log_quote(self, quote: Quote) -> None:
        self._conn.execute(
            """
            INSERT INTO quotes (symbol, bid, ask, last, timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                quote.symbol,
                float(quote.bid),
                float(quote.ask),
                float(quote.last),
                datetime.now(UTC).isoformat(),
            ),
        )
        self._conn.commit()

    def log_signal(self, symbol: str, signal_name: str, value: float) -> None:
        self._conn.execute(
            """
            INSERT INTO signals (symbol, signal_name, value, timestamp)
            VALUES (?, ?, ?, ?)
            """,
            (symbol, signal_name, value, datetime.now(UTC).isoformat()),
        )
        self._conn.commit()

    def log_order(
        self,
        order_result: OrderResult,
        symbol: str,
        side: str,
        quantity: float,
        paper_mode: bool,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO orders
                (symbol, side, quantity, status, filled_qty, avg_fill_price, paper_mode, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                symbol,
                side,
                float(quantity),
                order_result.status,
                float(order_result.filled_qty) if order_result.filled_qty is not None else None,
                float(order_result.avg_fill_price)
                if order_result.avg_fill_price is not None
                else None,
                int(paper_mode),
                datetime.now(UTC).isoformat(),
            ),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
