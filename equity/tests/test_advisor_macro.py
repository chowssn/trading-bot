"""Tests for equity/telegram/advisor.py's live macro snapshot (get_live_macro_snapshot)
— the advisor-side half of the macro monitoring extension. The alert-side half
(_check_macro_alerts) is covered in test_bot_macro_alerts.py.
"""

import unittest
from unittest.mock import MagicMock, patch

from equity.telegram import advisor as advisor_module
from equity.telegram.advisor import Advisor


def _make_advisor() -> Advisor:
    # thread_manager is a MagicMock — none of the methods under test touch
    # it, and a real ThreadManager() would open a sqlite file for no reason.
    return Advisor(api_key="test-key", thread_manager=MagicMock())


_FAKE_SNAPSHOT = {
    "treasury_curve": {
        "2Y": {"yield_pct": 4.10, "change_1d_bps": 3.2},
        "10Y": {"yield_pct": 4.35, "change_1d_bps": -1.5},
    },
    "fx": {
        "EURUSD=X": {"label": "EUR/USD", "price": 1.0850, "change_1d_pct": 0.25},
    },
    "commodities_extended": {
        "GC=F": {"label": "Gold", "price": 2650.10, "change_1d_pct": 1.1},
    },
    "commodities": {},
    "data_warnings": [],
}


class TestGetLiveMacroSnapshot(unittest.TestCase):
    def setUp(self):
        # Module-level cache is shared state — reset it between tests.
        advisor_module._macro_snapshot_cache["text"] = None
        advisor_module._macro_snapshot_cache["timestamp"] = 0.0

    @patch("equity.telegram.advisor.market_snapshot.fetch_market_snapshot", return_value=_FAKE_SNAPSHOT)
    def test_includes_available_sections(self, mock_fetch):
        result = _make_advisor().get_live_macro_snapshot()
        self.assertIn("2Y: 4.100% (+3.2bp)", result)
        self.assertIn("EUR/USD", result)
        self.assertIn("Gold", result)

    @patch("equity.telegram.advisor.market_snapshot.fetch_market_snapshot", return_value=_FAKE_SNAPSHOT)
    def test_caches_within_ttl(self, mock_fetch):
        adv = _make_advisor()
        adv.get_live_macro_snapshot()
        adv.get_live_macro_snapshot()
        self.assertEqual(mock_fetch.call_count, 1)

    @patch("equity.telegram.advisor.market_snapshot.fetch_market_snapshot", side_effect=Exception("boom"))
    def test_never_raises_on_failure(self, mock_fetch):
        self.assertEqual(_make_advisor().get_live_macro_snapshot(), "")

    @patch("equity.telegram.advisor.market_snapshot.fetch_market_snapshot", return_value={})
    def test_empty_snapshot_still_returns_header(self, mock_fetch):
        self.assertIn("LIVE MACRO DATA", _make_advisor().get_live_macro_snapshot())


class TestMacroProxyTickers(unittest.TestCase):
    # get_ticker_context()'s Section 7 (live macro context) gates purely on
    # `ticker.upper() in MACRO_PROXY_TICKERS` — get_ticker_context() itself
    # isn't unit tested anywhere in this repo (its other six sections hit
    # real yfinance/news calls), so this checks the gate's membership set
    # directly rather than a network-touching integration test.
    def test_contains_expected_symbols(self):
        for ticker in ("TLT", "GLD", "SLV", "URA", "TIP", "PPLT", "GC=F", "SI=F"):
            self.assertIn(ticker, advisor_module.MACRO_PROXY_TICKERS)

    def test_excludes_ordinary_equity_positions(self):
        self.assertNotIn("AAPL", advisor_module.MACRO_PROXY_TICKERS)
        self.assertNotIn("MSFT", advisor_module.MACRO_PROXY_TICKERS)


if __name__ == "__main__":
    unittest.main()
