"""Tests for equity/telegram/bot.py's ticker_news_/ticker_metrics_ callback
guards — the Telegram-callback half of the non-ticker-subject guard (the
advisor.get_ticker_context() half is covered in
test_advisor_ticker_validation.py).

These buttons are wired up with whatever "ticker" a monitoring item was
tagged with (see make_ticker_actions() call sites in bot.py), which can be
a thematic label like "COPPER" rather than a real yfinance ticker — see
NON_TICKER_SUBJECTS in equity/telegram/advisor.py. Clicking one must never
reach run_news_triage()/score_ticker() (and, transitively, yfinance) for
such a label.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from equity.telegram import bot


def _make_update_context(callback_data: str):
    query = MagicMock()
    query.answer = AsyncMock()
    query.data = callback_data
    query.message = MagicMock()
    query.message.chat_id = 12345
    query.message.reply_text = AsyncMock()

    update = MagicMock()
    update.callback_query = query
    # @authorized_only silently no-ops for any other user id — see
    # equity/telegram/bot.py's authorized_only().
    update.effective_user.id = bot.TELEGRAM_USER_ID

    context = MagicMock()
    context.bot = MagicMock()
    context.bot.send_message = AsyncMock()

    return update, context, query


class TestTickerNewsCallbackGuard(unittest.TestCase):
    def test_non_ticker_subject_skips_news_fetch(self):
        update, context, query = _make_update_context("ticker_news_COPPER")
        with patch.object(bot, "run_news_triage") as mock_triage:
            asyncio.run(bot.handle_callback(update, context))
        mock_triage.assert_not_called()
        query.message.reply_text.assert_awaited_once()
        self.assertIn("not a tradable ticker", query.message.reply_text.await_args.args[0])

    def test_real_ticker_still_runs_news_fetch(self):
        update, context, query = _make_update_context("ticker_news_TSLA")
        with patch.object(bot, "run_news_triage", return_value={}) as mock_triage, \
             patch.object(bot, "format_news_triage", return_value="text"), \
             patch.object(bot, "send_in_parts", new=AsyncMock()):
            asyncio.run(bot.handle_callback(update, context))
        mock_triage.assert_called_once_with(["TSLA"])


class TestTickerMetricsCallbackGuard(unittest.TestCase):
    def test_non_ticker_subject_skips_quality_score_fetch(self):
        update, context, query = _make_update_context("ticker_metrics_COPPER")
        with patch.object(bot, "score_ticker") as mock_score:
            asyncio.run(bot.handle_callback(update, context))
        mock_score.assert_not_called()
        query.message.reply_text.assert_awaited_once()
        self.assertIn("not a tradable ticker", query.message.reply_text.await_args.args[0])

    def test_real_ticker_still_runs_quality_score_fetch(self):
        update, context, query = _make_update_context("ticker_metrics_TSLA")
        with patch.object(bot, "score_ticker", return_value={"tier": "tier1"}) as mock_score:
            asyncio.run(bot.handle_callback(update, context))
        mock_score.assert_called_once_with("TSLA", bot.os.getenv("FMP_API_KEY"))


if __name__ == "__main__":
    unittest.main()
