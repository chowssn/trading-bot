"""Tests for equity/telegram/advisor.py's _is_valid_ticker() guard.

Monitoring items are parsed from free-text Claude synthesis output
(equity.brief.brief_synthesizer) and can carry a thematic/macro label —
e.g. "COPPER" for a copper/gold-ratio watch item — in the field
get_ticker_context()/bot.py's ticker_* callback handlers otherwise treat
as a real yfinance ticker. yfinance doesn't error on an unrecognized
symbol, it silently returns empty/None-filled data, so the guard has to
reject the label before any yfinance/quality_scorer call, not after.
"""

import unittest
from unittest.mock import MagicMock, patch

from equity.telegram.advisor import Advisor, NON_TICKER_SUBJECTS, _is_valid_ticker


def _make_advisor() -> Advisor:
    return Advisor(api_key="test-key", thread_manager=MagicMock())


class TestIsValidTicker(unittest.TestCase):
    def test_known_non_ticker_subject_is_invalid(self):
        self.assertFalse(_is_valid_ticker("COPPER"))

    def test_is_case_insensitive(self):
        self.assertFalse(_is_valid_ticker("copper"))

    def test_empty_or_none_is_invalid(self):
        self.assertFalse(_is_valid_ticker(""))
        self.assertFalse(_is_valid_ticker(None))

    def test_real_tickers_are_valid(self):
        for ticker in ["TSLA", "BRK.B", "^VIX", "GC=F", "EURUSD=X"]:
            self.assertTrue(_is_valid_ticker(ticker), ticker)

    def test_copper_is_in_non_ticker_subjects(self):
        self.assertIn("COPPER", NON_TICKER_SUBJECTS)


class TestGetTickerContextSkipsInvalidTicker(unittest.TestCase):
    """A non-ticker subject like "COPPER" must never reach yfinance or
    quality_scorer — those calls are skipped outright rather than run
    and produce garbage, per the guard's docstring.
    """

    def setUp(self):
        self._patches = [
            patch("yfinance.Ticker"),
            patch("equity.telegram.advisor.quality_scorer.score_ticker"),
            patch.object(Advisor, "get_regime_context", return_value=""),
        ]
        mocks = [p.start() for p in self._patches]
        for p in self._patches:
            self.addCleanup(p.stop)
        self.mock_ticker_cls, self.mock_score_ticker = mocks[0], mocks[1]

    def test_invalid_ticker_never_calls_yfinance_or_quality_scorer(self):
        adv = _make_advisor()
        with patch.object(adv, "_fetch_web_fundamentals", return_value={}):
            adv.get_ticker_context("COPPER")
        self.mock_ticker_cls.assert_not_called()
        self.mock_score_ticker.assert_not_called()

    def test_invalid_ticker_context_says_so_and_does_not_raise(self):
        adv = _make_advisor()
        with patch.object(adv, "_fetch_web_fundamentals", return_value={}):
            result = adv.get_ticker_context("COPPER")
        self.assertIn("monitoring subject, not a tradable ticker", result)

    def test_valid_ticker_still_calls_yfinance(self):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = None
        mock_ticker.info = {}
        mock_ticker.news = []
        self.mock_ticker_cls.return_value = mock_ticker
        self.mock_score_ticker.return_value = {"tier": "error"}

        adv = _make_advisor()
        with patch.object(adv, "_fetch_web_fundamentals", return_value={}):
            adv.get_ticker_context("TSLA")
        self.mock_ticker_cls.assert_called_with("TSLA")
        self.mock_score_ticker.assert_called_once()


if __name__ == "__main__":
    unittest.main()
