"""Tests for equity/telegram/advisor.py's live macro snapshot (get_live_macro_snapshot)
— the advisor-side half of the macro monitoring extension. The alert-side half
(_check_macro_alerts) is covered in test_bot_macro_alerts.py.

get_live_macro_snapshot() reads the shared `equity.data.price_cache`
singleton — these tests patch its `get`/`get_yield` methods rather than
hitting real yfinance/FRED.

TestGetMacroContext covers get_macro_context() (the web-searched fiscal/
monetary baseline for MACRO thread discussions) — patching
Advisor._web_search_and_extract() rather than the Anthropic client itself,
since that method (shared with _fetch_web_fundamentals()) is already
responsible for the actual search/extract/failure-handling mechanics.
"""

import time
import unittest
from unittest.mock import MagicMock, patch

from equity.telegram import advisor as advisor_module
from equity.telegram.advisor import Advisor


def _make_advisor() -> Advisor:
    # thread_manager is a MagicMock — none of the methods under test touch
    # it, and a real ThreadManager() would open a sqlite file for no reason.
    return Advisor(api_key="test-key", thread_manager=MagicMock())


_FAKE_CACHE = {
    "2Y": {"price": 4.10, "prev_close": 4.068, "change_1d_bps": 3.2, "is_yield": True},
    "10Y": {"price": 4.35, "prev_close": 4.365, "change_1d_bps": -1.5, "is_yield": True},
    "EURUSD=X": {"price": 1.0850, "prev_close": 1.0823, "change_1d_pct": 0.25, "is_yield": False},
    "GC=F": {"price": 2650.10, "prev_close": 2621.28, "change_1d_pct": 1.1, "is_yield": False},
}


def _fake_get(ticker):
    return _FAKE_CACHE.get(ticker)


def _fake_get_yield(tenor):
    return _FAKE_CACHE.get(tenor)


class TestGetLiveMacroSnapshot(unittest.TestCase):
    def setUp(self):
        # Module-level cache is shared state — reset it between tests.
        advisor_module._macro_snapshot_cache["text"] = None
        advisor_module._macro_snapshot_cache["timestamp"] = 0.0
        self._patches = (
            patch.object(advisor_module.price_cache, "get", side_effect=_fake_get),
            patch.object(advisor_module.price_cache, "get_yield", side_effect=_fake_get_yield),
        )
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()

    def test_includes_available_sections(self):
        result = _make_advisor().get_live_macro_snapshot()
        self.assertIn("2Y: 4.100% (+3.2bp)", result)
        self.assertIn("EUR/USD", result)
        self.assertIn("Gold", result)

    def test_caches_within_ttl(self):
        adv = _make_advisor()
        adv.get_live_macro_snapshot()
        with patch.object(advisor_module.price_cache, "get_yield", side_effect=AssertionError("should be cached")):
            adv.get_live_macro_snapshot()  # must not touch price_cache again

    def test_empty_cache_still_returns_header(self):
        with patch.object(advisor_module.price_cache, "get", return_value=None), \
             patch.object(advisor_module.price_cache, "get_yield", return_value=None):
            result = _make_advisor().get_live_macro_snapshot()
        self.assertIn("LIVE MACRO DATA", result)

    def test_never_raises_on_failure(self):
        with patch.object(advisor_module.price_cache, "get_yield", side_effect=Exception("boom")):
            self.assertEqual(_make_advisor().get_live_macro_snapshot(), "")


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


class TestGetMacroContext(unittest.TestCase):
    def test_runs_two_baseline_searches_and_combines_results(self):
        adv = _make_advisor()
        with patch.object(Advisor, "_web_search_and_extract", side_effect=["fiscal figures", "fed figures"]) as mock_search:
            result = adv.get_macro_context()
        self.assertEqual(mock_search.call_count, 2)
        self.assertIn("fiscal figures", result)
        self.assertIn("fed figures", result)
        self.assertIn("CURRENT MACRO DATA", result)

    def test_topic_over_three_chars_adds_third_search(self):
        adv = _make_advisor()
        with patch.object(Advisor, "_web_search_and_extract", return_value="figures") as mock_search:
            adv.get_macro_context(topic="unemployment rate")
        self.assertEqual(mock_search.call_count, 3)

    def test_short_topic_does_not_add_third_search(self):
        adv = _make_advisor()
        with patch.object(Advisor, "_web_search_and_extract", return_value="figures") as mock_search:
            adv.get_macro_context(topic="cpi")
        self.assertEqual(mock_search.call_count, 2)

    def test_caches_within_ttl(self):
        adv = _make_advisor()
        with patch.object(Advisor, "_web_search_and_extract", return_value="figures"):
            first = adv.get_macro_context()
        with patch.object(Advisor, "_web_search_and_extract", side_effect=AssertionError("should be cached")):
            second = adv.get_macro_context()
        self.assertEqual(first, second)

    def test_different_topics_cache_separately(self):
        adv = _make_advisor()
        with patch.object(Advisor, "_web_search_and_extract", return_value="A"):
            adv.get_macro_context(topic="cpi data")
        with patch.object(Advisor, "_web_search_and_extract", return_value="B"):
            result = adv.get_macro_context(topic="fed funds rate")
        self.assertIn("B", result)

    def test_returns_empty_string_when_all_searches_fail(self):
        adv = _make_advisor()
        with patch.object(Advisor, "_web_search_and_extract", return_value=""):
            self.assertEqual(adv.get_macro_context(), "")

    def test_never_raises_on_search_exception(self):
        adv = _make_advisor()
        with patch.object(Advisor, "_web_search_and_extract", side_effect=Exception("boom")):
            self.assertEqual(adv.get_macro_context(), "")

    def test_respects_shared_web_fundamentals_budget(self):
        adv = _make_advisor()
        adv._web_fundamentals_call_times = [time.time()] * advisor_module.WEB_FUNDAMENTALS_MAX_PER_HOUR
        with patch.object(Advisor, "_web_search_and_extract", side_effect=AssertionError("budget exhausted — should not search")):
            self.assertEqual(adv.get_macro_context(), "")


class TestOtherThreadDepth(unittest.TestCase):
    def test_same_day_gets_full_verbatim(self):
        verbatim, include_summary, chars, label = advisor_module._other_thread_depth(2.0)
        self.assertEqual(verbatim, 10)
        self.assertTrue(include_summary)
        self.assertIn("h ago", label)

    def test_yesterday_gets_moderate_verbatim(self):
        verbatim, _, _, label = advisor_module._other_thread_depth(30.0)
        self.assertEqual(verbatim, 5)
        self.assertEqual(label, "yesterday")

    def test_this_week_gets_light_verbatim(self):
        verbatim, _, _, label = advisor_module._other_thread_depth(4 * 24.0)
        self.assertEqual(verbatim, 2)

    def test_this_month_gets_summary_only(self):
        verbatim, include_summary, _, _ = advisor_module._other_thread_depth(15 * 24.0)
        self.assertEqual(verbatim, 0)
        self.assertTrue(include_summary)

    def test_past_30_days_is_omitted(self):
        verbatim, include_summary, _, _ = advisor_module._other_thread_depth(45 * 24.0)
        self.assertIsNone(verbatim)
        self.assertFalse(include_summary)


if __name__ == "__main__":
    unittest.main()
