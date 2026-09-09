"""Tests for equity/data/price_cache.py's after-hours price/staleness logic.

Regression coverage for two bugs:

1. Daily-bar age was measured against the UTC date. From 20:00 ET onward
   the UTC date has already rolled over, so an evening read of the current
   session's own close came back a full day older than it was ("24h ago"
   at 9 PM on the day of that close).
2. Outside market hours the daily bar IS the price, and nothing questioned
   how old it was — the equity staleness threshold (75h, sized for a
   weekend) accepts a bar several sessions stale, so an unpublished session
   silently showed an older close as if it were current.
"""

import datetime as dt
import unittest
from unittest.mock import patch

import pandas as pd
import pytz

from equity.data import price_cache as pc

ET = pytz.timezone("America/New_York")


def _frozen_datetime(utc_moment: dt.datetime):
    """A `datetime` stand-in whose `.now()` returns `utc_moment`."""

    class FrozenDateTime(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return utc_moment.astimezone(tz) if tz else utc_moment.replace(tzinfo=None)

    return FrozenDateTime


# 21:00 ET Tuesday 2026-09-08 — the UTC date has already rolled to Wednesday.
NINE_PM_ET_TUESDAY = dt.datetime(2026, 9, 9, 1, 0, tzinfo=pytz.utc)


class TestTradingDaysBetween(unittest.TestCase):
    def test_same_day_is_zero(self):
        d = dt.date(2026, 9, 8)
        self.assertEqual(pc._trading_days_between(d, d), 0)

    def test_friday_to_monday_is_one(self):
        self.assertEqual(pc._trading_days_between(dt.date(2026, 9, 4), dt.date(2026, 9, 7)), 1)

    def test_friday_to_wednesday_is_three(self):
        self.assertEqual(pc._trading_days_between(dt.date(2026, 9, 4), dt.date(2026, 9, 9)), 3)

    def test_weekend_days_are_not_counted(self):
        # Fri -> Sat/Sun adds nothing.
        self.assertEqual(pc._trading_days_between(dt.date(2026, 9, 4), dt.date(2026, 9, 6)), 0)

    def test_future_date_is_zero_not_negative(self):
        self.assertEqual(pc._trading_days_between(dt.date(2026, 9, 9), dt.date(2026, 9, 4)), 0)


class TestDailyBarAgeUsesEasternDate(unittest.TestCase):
    def test_same_session_close_read_at_9pm_is_not_a_day_old(self):
        with patch.object(pc, "datetime", _frozen_datetime(NINE_PM_ET_TUESDAY)):
            ctx = pc._get_session_context("AAPL", pd.Timestamp("2026-09-08 00:00:00"), 100.0, 99.0)
        self.assertEqual(ctx["data_age_hours"], 0)
        self.assertEqual(ctx["session_label"], "live")
        self.assertFalse(ctx["is_stale"])

    def test_trading_date_label_is_still_the_bar_date(self):
        with patch.object(pc, "datetime", _frozen_datetime(NINE_PM_ET_TUESDAY)):
            ctx = pc._get_session_context("AAPL", pd.Timestamp("2026-09-08 00:00:00"), 100.0, 99.0)
        self.assertEqual(ctx["exact_timestamp"], "Tue 2026-09-08 close")

    def test_prior_session_close_is_one_day_old(self):
        with patch.object(pc, "datetime", _frozen_datetime(NINE_PM_ET_TUESDAY)):
            ctx = pc._get_session_context("AAPL", pd.Timestamp("2026-09-07 00:00:00"), 100.0, 99.0)
        self.assertEqual(ctx["data_age_hours"], 24)


class TestGetCurrentPriceAndBarTime(unittest.TestCase):
    DAILY = pd.Series([99.0, 101.0], index=pd.to_datetime(["2026-09-04", "2026-09-08"]))

    def test_intraday_bar_is_preferred(self):
        intraday = pd.Series([102.5], index=pd.to_datetime(["2026-09-08 15:55:00"]))
        price, ts, forced = pc._get_current_price_and_bar_time("AAPL", self.DAILY, intraday, 99.0)
        self.assertEqual(price, 102.5)
        self.assertEqual(pd.Timestamp(ts), pd.Timestamp("2026-09-08 15:55:00"))
        self.assertFalse(forced)

    def test_unmoved_intraday_bar_falls_back_to_daily(self):
        # Intraday tick sitting exactly on prior close while the daily bar
        # has moved means the intraday bar hasn't updated.
        intraday = pd.Series([99.0], index=pd.to_datetime(["2026-09-08 15:55:00"]))
        price, ts, forced = pc._get_current_price_and_bar_time("AAPL", self.DAILY, intraday, 99.0)
        self.assertEqual(price, 101.0)
        self.assertEqual(pd.Timestamp(ts), pd.Timestamp("2026-09-08"))

    def test_empty_intraday_uses_daily_bar(self):
        with patch.object(pc, "datetime", _frozen_datetime(NINE_PM_ET_TUESDAY)):
            price, ts, forced = pc._get_current_price_and_bar_time(
                "AAPL", self.DAILY, pd.Series(dtype=float), 99.0
            )
        self.assertEqual(price, 101.0)
        self.assertFalse(forced)

    def test_stale_daily_bar_is_forced_stale(self):
        # Bar is 3 trading days old but only 72h — inside the 75h equity
        # threshold, so the age check alone would have called it fresh.
        daily = pd.Series([99.0, 101.0], index=pd.to_datetime(["2026-08-31", "2026-09-01"]))
        friday_morning = dt.datetime(2026, 9, 4, 13, 0, tzinfo=pytz.utc)
        with patch.object(pc, "datetime", _frozen_datetime(friday_morning)):
            _, ts, forced = pc._get_current_price_and_bar_time(
                "AAPL", daily, pd.Series(dtype=float), 99.0
            )
            unforced = pc._get_session_context("AAPL", ts, 101.0, 99.0, force_stale=False)
            forced_ctx = pc._get_session_context("AAPL", ts, 101.0, 99.0, force_stale=forced)
        self.assertTrue(forced)
        self.assertEqual(unforced["data_age_hours"], 72)
        self.assertFalse(unforced["is_stale"])   # what the old code concluded
        self.assertTrue(forced_ctx["is_stale"])  # what it should conclude
        self.assertIn("STALE", forced_ctx["session_label"])

    def test_extended_hours_instruments_are_exempt(self):
        # Futures/crypto trade on their own calendar and have their own
        # tighter thresholds — the trading-day check must not fire on them.
        ticker = next(iter(pc._EXTENDED_INTRADAY_SET))
        daily = pd.Series([99.0, 101.0], index=pd.to_datetime(["2026-08-31", "2026-09-01"]))
        friday_morning = dt.datetime(2026, 9, 4, 13, 0, tzinfo=pytz.utc)
        with patch.object(pc, "datetime", _frozen_datetime(friday_morning)):
            _, _, forced = pc._get_current_price_and_bar_time(
                ticker, daily, pd.Series(dtype=float), 99.0
            )
        self.assertFalse(forced)


class TestDailyOnlyTickers(unittest.TestCase):
    def test_move_index_skips_intraday(self):
        self.assertIn("^MOVE", pc.DAILY_ONLY_TICKERS)

    def test_tickers_with_real_intraday_data_are_not_listed(self):
        # ^VVIX and BKLN both return healthy 5m bars; listing them here would
        # needlessly downgrade them to a daily close.
        self.assertNotIn("^VVIX", pc.DAILY_ONLY_TICKERS)
        self.assertNotIn("BKLN", pc.DAILY_ONLY_TICKERS)

    def test_daily_only_tickers_are_still_in_the_fetch_list(self):
        # Skipping intraday must not mean skipping the ticker entirely — the
        # daily fetch still has to cover it.
        self.assertIn("^MOVE", pc.price_cache._build_ticker_list())


class TestForceStaleParameter(unittest.TestCase):
    def test_force_stale_overrides_a_fresh_age(self):
        ctx = pc._get_session_context("AAPL", pd.Timestamp("2026-09-08 00:00:00"), 100.0, 99.0, force_stale=True)
        self.assertTrue(ctx["is_stale"])

    def test_default_is_unforced(self):
        with patch.object(pc, "datetime", _frozen_datetime(NINE_PM_ET_TUESDAY)):
            ctx = pc._get_session_context("AAPL", pd.Timestamp("2026-09-08 00:00:00"), 100.0, 99.0)
        self.assertFalse(ctx["is_stale"])


if __name__ == "__main__":
    unittest.main()
