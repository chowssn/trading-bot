"""Tests for equity/screener/universe.py's ticker parsing and normalization.

Regression coverage for the failure that silently shrank the screener's
universe to 11 names: iShares switched share-class tickers from the
concatenated form ("BRKB") to a space-separated one ("BRK B"), and
`_parse_raw_holdings()` truncates the holdings table at the first ticker
that fails `TICKER_PATTERN`. Berkshire sits around row 12 by weight, so
every name below it was discarded.
"""

import unittest

import pandas as pd

from equity.screener import universe


class TestTickerPattern(unittest.TestCase):
    def test_accepts_plain_tickers(self):
        for ticker in ("A", "MU", "AAPL", "GOOGL", "XTSLA"):
            self.assertTrue(universe.TICKER_PATTERN.match(ticker), ticker)

    def test_accepts_space_separated_share_class(self):
        # The exact shape that truncated the table at 11 names.
        for ticker in ("BRK B", "HEI A", "BF B", "LEN B", "UHAL B"):
            self.assertTrue(universe.TICKER_PATTERN.match(ticker), ticker)

    def test_accepts_dot_and_hyphen_share_class(self):
        self.assertTrue(universe.TICKER_PATTERN.match("BRK.B"))
        self.assertTrue(universe.TICKER_PATTERN.match("BRK-B"))

    def test_still_rejects_futures_and_cash_rows(self):
        # These mark the real end of the holdings table and must keep failing,
        # otherwise truncation never happens and footer rows leak in.
        for ticker in ("FAU6", "ESU6", "", "-", "SOME LONG NAME"):
            self.assertFalse(universe.TICKER_PATTERN.match(ticker), ticker)


class TestNormalizeTicker(unittest.TestCase):
    def test_space_separator_becomes_hyphen(self):
        self.assertEqual(universe._normalize_ticker("BRK B"), "BRK-B")
        self.assertEqual(universe._normalize_ticker("HEI A"), "HEI-A")
        self.assertEqual(universe._normalize_ticker("UHAL B"), "UHAL-B")

    def test_dot_separator_becomes_hyphen(self):
        self.assertEqual(universe._normalize_ticker("BF.B"), "BF-B")

    def test_explicit_translation_still_wins(self):
        # Legacy concatenated forms stay mapped, so a feed revert is a no-op.
        self.assertEqual(universe._normalize_ticker("BRKB"), "BRK-B")
        self.assertEqual(universe._normalize_ticker("HEIA"), "HEI-A")

    def test_plain_ticker_passes_through(self):
        self.assertEqual(universe._normalize_ticker("AAPL"), "AAPL")

    def test_translate_ticker_is_the_public_alias(self):
        self.assertEqual(universe.translate_ticker("BRK B"), "BRK-B")
        self.assertEqual(universe.translate_ticker("AAPL"), "AAPL")


def _raw_holdings(tickers: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "Ticker": tickers,
        "Name": [f"{t} Inc" for t in tickers],
        "Sector": ["Information Technology"] * len(tickers),
        "Asset Class": ["Equity"] * len(tickers),
        "Weight (%)": [1.0] * len(tickers),
        "Market Value": [1_000.0] * len(tickers),
    })


class TestCleanHoldings(unittest.TestCase):
    def test_share_class_tickers_survive_and_are_normalized(self):
        df = universe._clean_holdings(_raw_holdings(["NVDA", "BRK B", "AAPL", "HEI A"]))
        self.assertEqual(df["ticker"].tolist(), ["NVDA", "BRK-B", "AAPL", "HEI-A"])

    def test_normalization_runs_before_the_validity_filter(self):
        # Regression: when normalization ran last, the space-form tickers were
        # already dropped by the no-spaces validity check and never reached it.
        df = universe._clean_holdings(_raw_holdings(["BRK B"]))
        self.assertEqual(df["ticker"].tolist(), ["BRK-B"])

    def test_non_equity_and_blacklisted_rows_are_dropped(self):
        raw = _raw_holdings(["NVDA", "USD", "CASH"])
        df = universe._clean_holdings(raw)
        self.assertEqual(df["ticker"].tolist(), ["NVDA"])

    def test_delisted_tickers_are_dropped(self):
        df = universe._clean_holdings(_raw_holdings(["NVDA", "HOLX"]))
        self.assertEqual(df["ticker"].tolist(), ["NVDA"])


class TestParseRawHoldings(unittest.TestCase):
    HEADER = "Ticker,Name,Sector,Asset Class,Market Value,Weight (%)\n"

    def test_does_not_truncate_on_share_class_ticker(self):
        csv_text = (
            "Fund metadata line\n"
            + self.HEADER
            + "NVDA,NVIDIA,Information Technology,Equity,100,1.0\n"
            + "BRK B,BERKSHIRE CLASS B,Financials,Equity,100,1.0\n"
            + "AAPL,APPLE,Information Technology,Equity,100,1.0\n"
        )
        df = universe._parse_raw_holdings(csv_text)
        self.assertEqual(len(df), 3)

    def test_still_truncates_at_futures_rows(self):
        csv_text = (
            "Fund metadata line\n"
            + self.HEADER
            + "NVDA,NVIDIA,Information Technology,Equity,100,1.0\n"
            + "BRK B,BERKSHIRE CLASS B,Financials,Equity,100,1.0\n"
            + "FAU6,RUSSELL FUTURES,-,Futures,100,1.0\n"
            + "ESU6,S&P FUTURES,-,Futures,100,1.0\n"
        )
        df = universe._parse_raw_holdings(csv_text)
        self.assertEqual(df["Ticker"].tolist(), ["NVDA", "BRK B"])


class TestMinExpectedGuard(unittest.TestCase):
    def test_threshold_is_far_above_the_truncated_count(self):
        # The bug produced 11 tickers; the guard has to reject that.
        self.assertGreater(universe.MIN_EXPECTED_TICKERS, 11)


if __name__ == "__main__":
    unittest.main()
