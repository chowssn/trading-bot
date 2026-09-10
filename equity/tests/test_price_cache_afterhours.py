"""Tests for equity/data/price_cache.py's after-hours pricing support.

Regression coverage for the gap this closes: after the 4 PM ET close, the
standard-ticker intraday fetch was gated behind `is_market_hours` (the
9:30-16:00 ET cash session) and simply wasn't attempted, so a post-close
read fell all the way back to the daily bar — a session behind — instead
of an after-hours print, until the next morning's fetch. See refresh()'s
docstring for the `prepost=True` / `_is_extended_hours_window()` fix and
the `official_close`/`afterhours_price`/`session_type` split this adds to
each cache entry.
"""

import datetime as dt
import unittest

import pandas as pd
import pytz

from equity.data import price_cache as pc

ET = pytz.timezone("America/New_York")


class TestBarSessionType(unittest.TestCase):
    def test_premarket_bar(self):
        bar = ET.localize(dt.datetime(2026, 9, 9, 8, 0))
        self.assertEqual(pc._bar_session_type(bar), "premarket")

    def test_regular_session_open(self):
        bar = ET.localize(dt.datetime(2026, 9, 9, 9, 30))
        self.assertEqual(pc._bar_session_type(bar), "regular")

    def test_regular_session_mid_day(self):
        bar = ET.localize(dt.datetime(2026, 9, 9, 12, 0))
        self.assertEqual(pc._bar_session_type(bar), "regular")

    def test_afterhours_bar(self):
        bar = ET.localize(dt.datetime(2026, 9, 9, 18, 0))
        self.assertEqual(pc._bar_session_type(bar), "afterhours")

    def test_regular_session_end_boundary_is_afterhours(self):
        # 16:00 sharp is the first after-hours bar, not the last regular one.
        bar = ET.localize(dt.datetime(2026, 9, 9, 16, 0))
        self.assertEqual(pc._bar_session_type(bar), "afterhours")

    def test_before_premarket_is_outside(self):
        bar = ET.localize(dt.datetime(2026, 9, 9, 2, 0))
        self.assertEqual(pc._bar_session_type(bar), "outside")

    def test_after_afterhours_is_outside(self):
        bar = ET.localize(dt.datetime(2026, 9, 9, 21, 0))
        self.assertEqual(pc._bar_session_type(bar), "outside")


class TestIsExtendedHoursWindow(unittest.TestCase):
    def test_weekday_within_window(self):
        now = ET.localize(dt.datetime(2026, 9, 9, 18, 0))  # Wednesday
        self.assertTrue(pc._is_extended_hours_window(now))

    def test_weekday_before_premarket(self):
        now = ET.localize(dt.datetime(2026, 9, 9, 2, 0))
        self.assertFalse(pc._is_extended_hours_window(now))

    def test_weekday_after_afterhours(self):
        now = ET.localize(dt.datetime(2026, 9, 9, 21, 0))
        self.assertFalse(pc._is_extended_hours_window(now))

    def test_saturday_is_never_extended_hours(self):
        now = ET.localize(dt.datetime(2026, 9, 12, 10, 0))  # Saturday
        self.assertFalse(pc._is_extended_hours_window(now))


class TestRegularSessionBars(unittest.TestCase):
    def test_filters_out_extended_hours_bars(self):
        idx = pd.to_datetime([
            "2026-09-09 08:00:00", "2026-09-09 09:30:00",
            "2026-09-09 15:55:00", "2026-09-09 18:00:00",
        ]).tz_localize(ET)
        series = pd.Series([99.0, 100.0, 101.0, 102.0], index=idx)
        result = pc._regular_session_bars(series)
        self.assertEqual(list(result), [100.0, 101.0])

    def test_empty_series_returns_empty(self):
        result = pc._regular_session_bars(pd.Series(dtype=float))
        self.assertEqual(len(result), 0)

    def test_no_regular_bars_returns_empty(self):
        idx = pd.to_datetime(["2026-09-09 18:00:00", "2026-09-09 19:00:00"]).tz_localize(ET)
        series = pd.Series([101.0, 102.0], index=idx)
        result = pc._regular_session_bars(series)
        self.assertEqual(len(result), 0)


class TestLatestExtendedHoursBar(unittest.TestCase):
    def test_afterhours_when_last_bar_is_afterhours(self):
        idx = pd.to_datetime([
            "2026-09-09 15:55:00", "2026-09-09 16:30:00", "2026-09-09 18:00:00",
        ]).tz_localize(ET)
        series = pd.Series([101.0, 100.5, 99.8], index=idx)
        price, session = pc._latest_extended_hours_bar(series)
        self.assertEqual(price, 99.8)
        self.assertEqual(session, "afterhours")

    def test_premarket_when_last_bar_is_premarket(self):
        idx = pd.to_datetime(["2026-09-09 08:00:00", "2026-09-09 09:00:00"]).tz_localize(ET)
        series = pd.Series([100.0, 100.5], index=idx)
        price, session = pc._latest_extended_hours_bar(series)
        self.assertEqual(price, 100.5)
        self.assertEqual(session, "premarket")

    def test_none_when_last_bar_is_regular(self):
        # A stale premarket bar earlier in the series must not resurface
        # once the regular session has bars of its own.
        idx = pd.to_datetime(["2026-09-09 08:00:00", "2026-09-09 10:00:00"]).tz_localize(ET)
        series = pd.Series([100.0, 101.0], index=idx)
        price, session = pc._latest_extended_hours_bar(series)
        self.assertIsNone(price)
        self.assertIsNone(session)

    def test_empty_series_returns_none(self):
        price, session = pc._latest_extended_hours_bar(pd.Series(dtype=float))
        self.assertIsNone(price)
        self.assertIsNone(session)

    def test_naive_index_treated_as_et(self):
        idx = pd.to_datetime(["2026-09-09 18:00:00"])  # naive
        series = pd.Series([99.8], index=idx)
        price, session = pc._latest_extended_hours_bar(series)
        self.assertEqual(price, 99.8)
        self.assertEqual(session, "afterhours")


if __name__ == "__main__":
    unittest.main()
