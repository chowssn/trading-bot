"""Tests for equity/telegram/advisor.py's explicit-refresh support —
invalidate_ticker_cache(), _web_fundamentals_budget_available(), and that
chat() never invalidates on its own (/refresh TICKER is the sole path).
The Claude client, ThreadManager and get_ticker_context() are mocked.
"""

import unittest
from unittest.mock import MagicMock, patch

from equity.telegram.advisor import WEB_FUNDAMENTALS_MAX_PER_HOUR, Advisor


def _make_advisor() -> Advisor:
    thread_manager = MagicMock()
    thread_manager.get_messages_for_api.return_value = [{"role": "user", "content": "q"}]
    return Advisor(api_key="test-key", thread_manager=thread_manager)


def _response(text: str = "ok"):
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.stop_reason = "end_turn"
    response.content = [block]
    response.usage = MagicMock(input_tokens=10, output_tokens=10)
    return response


class TestInvalidateTickerCache(unittest.TestCase):
    def test_clears_only_matching_ticker(self):
        adv = _make_advisor()
        adv._web_fundamentals_cache = {
            "CEG_2026-10-14": {"a": "x"},
            "CEG_2026-10-13": {"a": "y"},
            "CEGX_2026-10-14": {"a": "z"},
            "APP_2026-10-14": {"a": "w"},
        }
        adv.invalidate_ticker_cache("ceg")
        self.assertEqual(
            set(adv._web_fundamentals_cache), {"CEGX_2026-10-14", "APP_2026-10-14"}
        )

    def test_no_entries_is_noop(self):
        adv = _make_advisor()
        adv.invalidate_ticker_cache("CEG")
        self.assertEqual(adv._web_fundamentals_cache, {})


class TestBudgetAvailable(unittest.TestCase):
    def test_peek_does_not_consume_budget(self):
        adv = _make_advisor()
        self.assertTrue(adv._web_fundamentals_budget_available())
        self.assertEqual(adv._web_fundamentals_call_times, [])

    def test_exhausted_budget_reports_unavailable(self):
        adv = _make_advisor()
        with patch("equity.telegram.advisor.time.time", return_value=1000.0):
            adv._web_fundamentals_call_times = [999.0] * WEB_FUNDAMENTALS_MAX_PER_HOUR
            self.assertFalse(adv._web_fundamentals_budget_available())

    def test_expired_calls_do_not_count(self):
        adv = _make_advisor()
        with patch("equity.telegram.advisor.time.time", return_value=10000.0):
            adv._web_fundamentals_call_times = [1.0] * WEB_FUNDAMENTALS_MAX_PER_HOUR
            self.assertTrue(adv._web_fundamentals_budget_available())


class TestChatDoesNotAutoRefresh(unittest.TestCase):
    def test_refresh_wording_leaves_cache_and_prompt_untouched(self):
        adv = _make_advisor()
        adv._web_fundamentals_cache = {"CEG_2026-10-14": {"a": "cached"}}
        with patch.object(adv, "get_ticker_context") as mock_ctx, \
             patch.object(adv.client.messages, "create", return_value=_response()) as mock_create:
            adv.chat("ticker_CEG", "refresh — give me the latest", "BASE PROMPT", "CEG")

        mock_ctx.assert_not_called()
        self.assertIn("CEG_2026-10-14", adv._web_fundamentals_cache)
        self.assertEqual(mock_create.call_args.kwargs["system"], "BASE PROMPT")


if __name__ == "__main__":
    unittest.main()
