"""Tests for equity/telegram/advisor.py's web-search gap-filling layer
(_web_search_and_extract, _fetch_web_fundamentals, _web_fundamentals_budget_ok)
— the Claude client is mocked throughout, so nothing here hits the real
Anthropic API or does an actual web search.
"""

import time
import unittest
from unittest.mock import MagicMock, patch

from equity.telegram import advisor as advisor_module
from equity.telegram.advisor import Advisor, WEB_FUNDAMENTALS_MAX_PER_HOUR


def _make_advisor() -> Advisor:
    return Advisor(api_key="test-key", thread_manager=MagicMock())


def _text_block(text: str):
    block = MagicMock()
    block.text = text
    return block


class TestWebSearchAndExtract(unittest.TestCase):
    def test_returns_extracted_text_on_success(self):
        adv = _make_advisor()
        search_response = MagicMock(content=[_text_block("raw search material")])
        extract_response = MagicMock(content=[_text_block("- Market cap: $500M (source, 2026-09-01)")])
        with patch.object(adv.client.messages, "create", side_effect=[search_response, extract_response]) as mock_create:
            result = adv._web_search_and_extract("query", "extract this")
        self.assertEqual(result, "- Market cap: $500M (source, 2026-09-01)")
        self.assertEqual(mock_create.call_count, 2)

    def test_empty_search_result_skips_extraction_call(self):
        adv = _make_advisor()
        search_response = MagicMock(content=[])
        with patch.object(adv.client.messages, "create", return_value=search_response) as mock_create:
            result = adv._web_search_and_extract("query", "extract this")
        self.assertEqual(result, "")
        mock_create.assert_called_once()  # only the search call, no extraction

    def test_api_exception_returns_empty_string_not_raise(self):
        adv = _make_advisor()
        with patch.object(adv.client.messages, "create", side_effect=RuntimeError("boom")):
            result = adv._web_search_and_extract("query", "extract this")  # must not raise
        self.assertEqual(result, "")

    def test_extraction_prompt_includes_rules_and_source(self):
        adv = _make_advisor()
        search_response = MagicMock(content=[_text_block("raw material")])
        extract_response = MagicMock(content=[_text_block("result")])
        with patch.object(adv.client.messages, "create", side_effect=[search_response, extract_response]) as mock_create:
            adv._web_search_and_extract("my query", "extract X")
        second_call_kwargs = mock_create.call_args_list[1].kwargs
        prompt = second_call_kwargs["messages"][0]["content"]
        self.assertIn("extract X", prompt)
        self.assertIn("raw material", prompt)
        self.assertIn("not found", prompt)
        self.assertIn("Never estimate or infer", prompt)


class TestWebFundamentalsBudget(unittest.TestCase):
    def test_allows_calls_up_to_the_cap(self):
        adv = _make_advisor()
        for _ in range(WEB_FUNDAMENTALS_MAX_PER_HOUR):
            self.assertTrue(adv._web_fundamentals_budget_ok())

    def test_denies_once_cap_exceeded(self):
        adv = _make_advisor()
        for _ in range(WEB_FUNDAMENTALS_MAX_PER_HOUR):
            adv._web_fundamentals_budget_ok()
        self.assertFalse(adv._web_fundamentals_budget_ok())

    def test_old_calls_age_out_of_the_window(self):
        adv = _make_advisor()
        # Fill the budget with calls that are already outside the 1h window.
        adv._web_fundamentals_call_times = [time.time() - 3700] * WEB_FUNDAMENTALS_MAX_PER_HOUR
        self.assertTrue(adv._web_fundamentals_budget_ok())


class TestFetchWebFundamentals(unittest.TestCase):
    def test_runs_four_searches_and_returns_keyed_results(self):
        adv = _make_advisor()
        with patch.object(adv, "_web_search_and_extract", return_value="some fact") as mock_search:
            result = adv._fetch_web_fundamentals("TSLA", "Tesla")
        self.assertEqual(mock_search.call_count, 4)
        self.assertEqual(
            set(result.keys()),
            {"fundamentals", "earnings_consensus", "recent_events", "competitive"},
        )

    def test_one_search_failing_does_not_drop_the_others(self):
        adv = _make_advisor()

        def _side_effect(query, extract_prompt, max_tokens=400):
            if "competitors" in query:
                raise RuntimeError("boom")
            return "fact"

        with patch.object(adv, "_web_search_and_extract", side_effect=_side_effect):
            result = adv._fetch_web_fundamentals("TSLA", "Tesla")
        self.assertNotIn("competitive", result)
        self.assertEqual(len(result), 3)

    def test_empty_extraction_result_omitted_from_dict(self):
        adv = _make_advisor()
        with patch.object(adv, "_web_search_and_extract", return_value=""):
            result = adv._fetch_web_fundamentals("TSLA", "Tesla")
        self.assertEqual(result, {})

    def test_over_budget_returns_empty_dict_without_searching(self):
        adv = _make_advisor()
        adv._web_fundamentals_call_times = [time.time()] * WEB_FUNDAMENTALS_MAX_PER_HOUR
        with patch.object(adv, "_web_search_and_extract") as mock_search:
            result = adv._fetch_web_fundamentals("TSLA", "Tesla")
        self.assertEqual(result, {})
        mock_search.assert_not_called()

    def test_falls_back_to_ticker_when_no_company_name(self):
        adv = _make_advisor()
        with patch.object(adv, "_web_search_and_extract", return_value="fact") as mock_search:
            adv._fetch_web_fundamentals("TSLA", "")
        queries = [call.args[0] for call in mock_search.call_args_list]
        self.assertTrue(any("TSLA" in q for q in queries))


class TestFetchWebFundamentalsCache(unittest.TestCase):
    """_fetch_web_fundamentals()'s per-ticker-per-day cache — added so a
    second /discuss TICKER open the same day reuses the morning's 4-search
    batch instead of re-running it (~$0.05/open saved). Same pattern as
    TestGetMacroContext's cache tests in test_advisor_macro.py.
    """

    def test_second_call_same_ticker_same_day_is_cached(self):
        adv = _make_advisor()
        with patch.object(adv, "_web_search_and_extract", return_value="fact"):
            first = adv._fetch_web_fundamentals("TSLA", "Tesla")
        with patch.object(adv, "_web_search_and_extract", side_effect=AssertionError("should be cached")):
            second = adv._fetch_web_fundamentals("TSLA", "Tesla")
        self.assertEqual(first, second)

    def test_different_ticker_not_served_from_cache(self):
        adv = _make_advisor()
        with patch.object(adv, "_web_search_and_extract", return_value="tsla fact"):
            adv._fetch_web_fundamentals("TSLA", "Tesla")
        with patch.object(adv, "_web_search_and_extract", return_value="aapl fact") as mock_search:
            result = adv._fetch_web_fundamentals("AAPL", "Apple")
        mock_search.assert_called()
        self.assertEqual(result["fundamentals"], "aapl fact")

    def test_different_day_not_served_from_cache(self):
        adv = _make_advisor()
        with patch.object(adv, "_web_search_and_extract", return_value="fact"):
            adv._fetch_web_fundamentals("TSLA", "Tesla")
        yesterday_key = f"TSLA_{advisor_module.date.today().isoformat()}"
        self.assertIn(yesterday_key, adv._web_fundamentals_cache)
        with patch.object(advisor_module, "date") as mock_date:
            mock_date.today.return_value = advisor_module.date(2099, 1, 1)
            with patch.object(adv, "_web_search_and_extract", return_value="fresh fact") as mock_search:
                result = adv._fetch_web_fundamentals("TSLA", "Tesla")
        mock_search.assert_called()
        self.assertEqual(result["fundamentals"], "fresh fact")

    def test_empty_result_is_not_cached(self):
        adv = _make_advisor()
        with patch.object(adv, "_web_search_and_extract", return_value=""):
            adv._fetch_web_fundamentals("TSLA", "Tesla")
        with patch.object(adv, "_web_search_and_extract", return_value="fact") as mock_search:
            result = adv._fetch_web_fundamentals("TSLA", "Tesla")
        mock_search.assert_called()
        self.assertEqual(result["fundamentals"], "fact")

    def test_over_budget_result_is_not_cached(self):
        adv = _make_advisor()
        adv._web_fundamentals_call_times = [time.time()] * advisor_module.WEB_FUNDAMENTALS_MAX_PER_HOUR
        with patch.object(adv, "_web_search_and_extract", side_effect=AssertionError("over budget — should not search")):
            adv._fetch_web_fundamentals("TSLA", "Tesla")
        adv._web_fundamentals_call_times = []
        with patch.object(adv, "_web_search_and_extract", return_value="fact") as mock_search:
            result = adv._fetch_web_fundamentals("TSLA", "Tesla")
        mock_search.assert_called()
        self.assertEqual(result["fundamentals"], "fact")


class TestGetTickerContextWebSection(unittest.TestCase):
    """get_ticker_context()'s integration of the web-fundamentals section.

    Mocks every network-touching dependency get_ticker_context() has —
    not just _fetch_web_fundamentals — since this test cares only about
    where the web section lands in the assembled context, not about
    yfinance/the quality scorer/regime snapshot themselves (those, and
    their own failure modes, belong to their own tests elsewhere).
    """

    def setUp(self):
        self._patches = [
            patch("yfinance.Ticker"),
            patch("equity.telegram.advisor.quality_scorer.score_ticker",
                  return_value={"tier": "error"}),
            patch.object(Advisor, "get_regime_context", return_value=""),
        ]
        mocks = [p.start() for p in self._patches]
        for p in self._patches:
            self.addCleanup(p.stop)
        mock_ticker_cls = mocks[0]
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = None
        mock_ticker.info = {}
        mock_ticker.news = []
        mock_ticker_cls.return_value = mock_ticker

    def test_web_data_present_is_labeled_and_included(self):
        adv = _make_advisor()
        with patch.object(adv, "_fetch_web_fundamentals", return_value={"fundamentals": "Market cap: $1B"}):
            result = adv.get_ticker_context("TSLA")
        self.assertIn("WEB-SEARCHED FUNDAMENTALS", result)
        self.assertIn("Market cap: $1B", result)

    def test_empty_web_data_shows_unavailable_note_not_fabrication(self):
        adv = _make_advisor()
        with patch.object(adv, "_fetch_web_fundamentals", return_value={}):
            result = adv.get_ticker_context("TSLA")
        self.assertIn("WEB-SEARCHED FUNDAMENTALS: unavailable", result)
        self.assertIn("Do not estimate", result)

    def test_web_fundamentals_exception_does_not_break_other_sections(self):
        adv = _make_advisor()
        with patch.object(adv, "_fetch_web_fundamentals", side_effect=RuntimeError("boom")):
            result = adv.get_ticker_context("TSLA")  # must not raise
        self.assertIn("WEB-SEARCHED FUNDAMENTALS: unavailable", result)
        self.assertIn("PRICE & TECHNICALS", result)


if __name__ == "__main__":
    unittest.main()
