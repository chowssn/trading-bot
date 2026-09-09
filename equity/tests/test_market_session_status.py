"""Tests for equity/brief/market_snapshot.py's _get_market_session_status()
— the real-time "is that index's exchange open right now" indicator shown
next to international indices (distinct from price_cache's is_stale/
session_label, which answers "how old is the data we have" — see that
function's docstring for why both are shown together)."""

import unittest
from datetime import datetime
from unittest.mock import patch

import pytz

from equity.brief.market_snapshot import _get_market_session_status


def _fake_now(year, month, day, hour, minute, tz_name="UTC"):
    tz = pytz.timezone(tz_name)
    return tz.localize(datetime(year, month, day, hour, minute))


class TestGetMarketSessionStatus(unittest.TestCase):
    def test_unknown_ticker_returns_empty_string(self):
        self.assertEqual(_get_market_session_status("^UNKNOWN"), "")

    @patch("equity.brief.market_snapshot.datetime")
    def test_weekend_is_closed(self, mock_datetime):
        # 2026-09-06 is a Sunday.
        mock_datetime.now.return_value = _fake_now(2026, 9, 6, 3, 0)
        self.assertIn("closed", _get_market_session_status("^N225"))

    @patch("equity.brief.market_snapshot.datetime")
    def test_during_session_is_live(self, mock_datetime):
        # 2026-09-07 is a Monday. Tokyo trades 09:00-15:30 JST = 00:00-06:30 UTC.
        mock_datetime.now.return_value = _fake_now(2026, 9, 7, 2, 0)
        self.assertIn("live", _get_market_session_status("^N225"))

    @patch("equity.brief.market_snapshot.datetime")
    def test_before_open_shows_countdown(self, mock_datetime):
        # 22:00 UTC Monday is 07:00 JST Tuesday — before Tokyo's 09:00 open.
        mock_datetime.now.return_value = _fake_now(2026, 9, 7, 22, 0)
        result = _get_market_session_status("^N225")
        self.assertIn("opens in", result)

    @patch("equity.brief.market_snapshot.datetime")
    def test_after_close_is_closed(self, mock_datetime):
        # 08:00 UTC Monday is 17:00 JST — after Tokyo's 15:30 close.
        mock_datetime.now.return_value = _fake_now(2026, 9, 7, 8, 0)
        self.assertIn("closed", _get_market_session_status("^N225"))


if __name__ == "__main__":
    unittest.main()
