"""Tests for storage/db.py using an in-memory SQLite database."""

import unittest
from datetime import datetime, timezone
from decimal import Decimal

from execution.broker_adapter import OrderResult, Quote
from storage.db import Database


class TestDatabase(unittest.TestCase):
    def setUp(self):
        self.db = Database(db_path=":memory:")

    def tearDown(self):
        self.db.close()

    def test_tables_created(self):
        cursor = self.db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
        table_names = {row[0] for row in cursor.fetchall()}
        self.assertIn("quotes", table_names)
        self.assertIn("signals", table_names)
        self.assertIn("orders", table_names)

    def test_log_quote(self):
        quote = Quote(
            bid=Decimal("100.5"),
            ask=Decimal("100.7"),
            last=Decimal("100.6"),
            timestamp=datetime.now(timezone.utc),
            symbol="BTC-USDC",
        )
        self.db.log_quote(quote)

        cursor = self.db._conn.execute(
            "SELECT symbol, bid, ask, last FROM quotes"
        )
        row = cursor.fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], "BTC-USDC")
        self.assertEqual(row[1], 100.5)
        self.assertEqual(row[2], 100.7)
        self.assertEqual(row[3], 100.6)

    def test_log_signal(self):
        self.db.log_signal("BTC-USDC", "momentum", 1.0)

        cursor = self.db._conn.execute(
            "SELECT symbol, signal_name, value FROM signals"
        )
        row = cursor.fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], "BTC-USDC")
        self.assertEqual(row[1], "momentum")
        self.assertEqual(row[2], 1.0)

    def test_log_order(self):
        order_result = OrderResult(
            order_id="abc123",
            status="filled",
            filled_qty=Decimal("1.0"),
            avg_fill_price=Decimal("100.6"),
        )
        self.db.log_order(
            order_result,
            symbol="BTC-USDC",
            side="buy",
            quantity=1.0,
            paper_mode=True,
        )

        cursor = self.db._conn.execute(
            "SELECT symbol, side, status, paper_mode FROM orders"
        )
        row = cursor.fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], "BTC-USDC")
        self.assertEqual(row[1], "buy")
        self.assertEqual(row[2], "filled")
        self.assertEqual(row[3], 1)

    def test_multiple_quotes(self):
        for _ in range(3):
            quote = Quote(
                bid=Decimal("100.5"),
                ask=Decimal("100.7"),
                last=Decimal("100.6"),
                timestamp=datetime.now(timezone.utc),
                symbol="BTC-USDC",
            )
            self.db.log_quote(quote)

        cursor = self.db._conn.execute(
            "SELECT COUNT(*) FROM quotes WHERE symbol = 'BTC-USDC'"
        )
        count = cursor.fetchone()[0]
        self.assertEqual(count, 3)

    def test_indexes_exist(self):
        cursor = self.db._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'index' AND tbl_name = 'quotes'"
        )
        index_sqls = [row[0] for row in cursor.fetchall() if row[0]]
        self.assertTrue(len(index_sqls) >= 1)
        self.assertTrue(
            any("symbol" in sql and "timestamp" in sql for sql in index_sqls)
        )


if __name__ == "__main__":
    unittest.main()
