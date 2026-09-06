"""Tests for equity/telegram/bot.py's intraday macro alert logic (Treasury
yields, FX, commodities) — the alert-side half of the macro monitoring
extension. The advisor-side half (get_live_macro_snapshot) is covered in
test_advisor_macro.py.
"""

import unittest
from unittest.mock import patch

from equity.telegram import bot


class TestLevelCrossed(unittest.TestCase):
    def test_crossed_upward(self):
        self.assertTrue(bot._level_crossed(prev=4.45, curr=4.55, level=4.5))

    def test_crossed_downward(self):
        self.assertTrue(bot._level_crossed(prev=4.55, curr=4.45, level=4.5))

    def test_not_crossed(self):
        self.assertFalse(bot._level_crossed(prev=4.40, curr=4.48, level=4.5))

    def test_landing_exactly_on_level_counts_as_crossed(self):
        self.assertTrue(bot._level_crossed(prev=4.45, curr=4.50, level=4.5))


class TestFxContextNote(unittest.TestCase):
    def test_known_pair_positive(self):
        self.assertIn("EUR strengthening", bot._fx_context_note("EURUSD=X", 0.6))

    def test_known_pair_negative(self):
        self.assertIn("JPY strengthening", bot._fx_context_note("USDJPY=X", -0.6))

    def test_unknown_pair_falls_back(self):
        self.assertIn("check commodity and EM exposure", bot._fx_context_note("AUDUSD=X", 0.6))


class TestCommodityContextNote(unittest.TestCase):
    def test_known_commodity(self):
        self.assertIn("Gold surging", bot._commodity_context_note("GC=F", 2.0))

    def test_unknown_commodity_falls_back(self):
        self.assertIn("check portfolio exposure", bot._commodity_context_note("PL=F", 2.0))


# Crafted to trip: a >8bp 10Y move + a 4.5% level breach, a >0.5% EUR/USD
# move, a sub-threshold DXY move (should NOT alert), and a >1.5% gold move.
_FAKE_SNAPSHOT = {
    "treasury_curve": {
        "10Y": {"yield_pct": 4.58, "change_1d_bps": 9.0},
    },
    "fx": {
        "EURUSD=X": {"label": "EUR/USD", "price": 1.09, "change_1d_pct": 0.7},
    },
    "commodities": {
        "DX-Y.NYB": {"price": 104.0, "change_1d_pct": 0.1},
    },
    "commodities_extended": {
        "GC=F": {"label": "Gold", "price": 2700.0, "change_1d_pct": 2.0},
    },
    "data_warnings": [],
}


class TestCheckMacroAlerts(unittest.TestCase):
    def setUp(self):
        bot._alerted_today.clear()

    @patch("equity.telegram.bot.fetch_market_snapshot", return_value=_FAKE_SNAPSHOT)
    def test_detects_yield_move_and_level_breach(self, mock_fetch):
        types = {a["type"] for a in bot._check_macro_alerts("2026-09-06")}
        self.assertIn("macro_yield", types)
        self.assertIn("macro_yield_level", types)

    @patch("equity.telegram.bot.fetch_market_snapshot", return_value=_FAKE_SNAPSHOT)
    def test_detects_fx_move_above_threshold(self, mock_fetch):
        fx_alerts = [a for a in bot._check_macro_alerts("2026-09-06") if a["type"] == "macro_fx"]
        self.assertEqual(len(fx_alerts), 1)
        self.assertEqual(fx_alerts[0]["ticker"], "EUR/USD")

    @patch("equity.telegram.bot.fetch_market_snapshot", return_value=_FAKE_SNAPSHOT)
    def test_dxy_below_threshold_not_alerted(self, mock_fetch):
        alerts = bot._check_macro_alerts("2026-09-06")
        self.assertFalse(any(a["ticker"] == "DXY Dollar Index" for a in alerts))

    @patch("equity.telegram.bot.fetch_market_snapshot", return_value=_FAKE_SNAPSHOT)
    def test_detects_commodity_move(self, mock_fetch):
        comm_alerts = [a for a in bot._check_macro_alerts("2026-09-06") if a["type"] == "macro_commodity"]
        self.assertEqual(len(comm_alerts), 1)
        self.assertEqual(comm_alerts[0]["ticker"], "Gold")

    @patch("equity.telegram.bot.fetch_market_snapshot", return_value=_FAKE_SNAPSHOT)
    def test_dedup_skips_already_alerted_key(self, mock_fetch):
        today = "2026-09-06"
        bot._alerted_today[f"yield_10Y_{today}"] = True
        alerts = bot._check_macro_alerts(today)
        self.assertFalse(any(a["type"] == "macro_yield" for a in alerts))

    @patch("equity.telegram.bot.fetch_market_snapshot", side_effect=Exception("boom"))
    def test_snapshot_failure_returns_empty_list(self, mock_fetch):
        self.assertEqual(bot._check_macro_alerts("2026-09-06"), [])


if __name__ == "__main__":
    unittest.main()
