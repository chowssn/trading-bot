"""Tests for equity/data/price_cache.py's pure logic: the tenor->cache-key
mapping (2Y/20Y resolve to their FRED series id, not a yfinance ticker —
see the module docstring) and the coverage report. The actual refresh()
(real yfinance/FRED calls) isn't exercised here — see
equity/tests/test_bot_macro_alerts.py and test_advisor_macro.py, which
patch PriceCache.get/get_yield rather than hitting the network.
"""

import time
import unittest
from datetime import datetime, timedelta

import pandas as pd
import pytz

from equity.data.price_cache import PriceCache, TENOR_TO_CACHE_KEY, _YIELD_SCALE, _get_session_context


class TestTenorToCacheKey(unittest.TestCase):
    def test_yfinance_tenors_map_to_their_ticker(self):
        self.assertEqual(TENOR_TO_CACHE_KEY["3M"], "^IRX")
        self.assertEqual(TENOR_TO_CACHE_KEY["5Y"], "^FVX")
        self.assertEqual(TENOR_TO_CACHE_KEY["10Y"], "^TNX")
        self.assertEqual(TENOR_TO_CACHE_KEY["30Y"], "^TYX")

    def test_fred_only_tenors_map_to_their_series_id(self):
        # 2Y/20Y aren't on yfinance at all — see module docstring.
        self.assertEqual(TENOR_TO_CACHE_KEY["2Y"], "DGS2")
        self.assertEqual(TENOR_TO_CACHE_KEY["20Y"], "DGS20")


class TestYieldScale(unittest.TestCase):
    def test_no_scaling_currently_applied(self):
        # Confirmed empirically 2026-09-06: yfinance no longer quotes
        # ^TNX/^TYX at 10x the actual yield. If this ever regresses, a 10Y
        # print like "47.8%" from get_yield('10Y') is the tell — see the
        # comment on _YIELD_SCALE in price_cache.py.
        for ticker in ("^IRX", "^FVX", "^TNX", "^TYX"):
            self.assertEqual(_YIELD_SCALE[ticker], 1.0)


class TestGetYield(unittest.TestCase):
    def test_resolves_tenor_through_cache_key(self):
        cache = PriceCache()
        cache._cache = {"DGS2": {"price": 4.1, "is_yield": True}}
        cache._last_fetch = time.time()  # fresh — refresh() is a no-op
        result = cache.get_yield("2Y")
        self.assertEqual(result["price"], 4.1)

    def test_unknown_tenor_returns_none(self):
        cache = PriceCache()
        cache._last_fetch = time.time()
        self.assertIsNone(cache.get_yield("15Y"))


class TestGetChange1d(unittest.TestCase):
    def test_yield_ticker_uses_bps_field(self):
        cache = PriceCache()
        cache._cache = {"^TNX": {"price": 4.5, "change_1d_bps": 3.0, "is_yield": True}}
        cache._last_fetch = time.time()
        self.assertEqual(cache.get_change_1d("^TNX"), 3.0)

    def test_regular_ticker_uses_pct_field(self):
        cache = PriceCache()
        cache._cache = {"SPY": {"price": 500.0, "change_1d_pct": 0.5, "is_yield": False}}
        cache._last_fetch = time.time()
        self.assertEqual(cache.get_change_1d("SPY"), 0.5)

    def test_missing_ticker_returns_none(self):
        cache = PriceCache()
        cache._last_fetch = time.time()
        self.assertIsNone(cache.get_change_1d("NOPE"))


class TestCoverageReport(unittest.TestCase):
    def test_never_fetched_reports_as_such(self):
        cache = PriceCache()
        self.assertIn("never fetched", cache.coverage_report())

    def test_reports_missing_tickers(self):
        cache = PriceCache()
        cache._all_tickers = ["SPY", "QQQ", "^MOVE"]
        cache._cache = {"SPY": {}, "QQQ": {}}
        cache._last_fetch = time.time()
        report = cache.coverage_report()
        self.assertIn("2/3", report)
        self.assertIn("^MOVE", report)


class TestGetSessionContext(unittest.TestCase):
    def test_none_timestamp_is_unknown_and_stale(self):
        ctx = _get_session_context("SPY", None, 500.0, 495.0)
        self.assertEqual(ctx["session_label"], "unknown")
        self.assertTrue(ctx["is_stale"])
        self.assertEqual(ctx["instrument_type"], "unknown")

    def test_recent_equity_bar_is_live(self):
        recent = pd.Timestamp(datetime.now(pytz.utc) - timedelta(minutes=5))
        ctx = _get_session_context("SPY", recent, 500.0, 495.0)
        self.assertEqual(ctx["session_label"], "live")
        self.assertFalse(ctx["is_stale"])
        self.assertEqual(ctx["instrument_type"], "equity")

    def test_crypto_uses_4h_threshold(self):
        five_hours_ago = pd.Timestamp(datetime.now(pytz.utc) - timedelta(hours=5))
        ctx = _get_session_context("BTC-USD", five_hours_ago, 90000.0, 89000.0)
        self.assertEqual(ctx["instrument_type"], "crypto")
        self.assertTrue(ctx["is_stale"])  # 5h > crypto's 4h threshold (24/7, no weekend gap)

    def test_crypto_within_4h_is_not_stale(self):
        three_hours_ago = pd.Timestamp(datetime.now(pytz.utc) - timedelta(hours=3))
        ctx = _get_session_context("BTC-USD", three_hours_ago, 90000.0, 89000.0)
        self.assertFalse(ctx["is_stale"])  # 3h < crypto's 4h threshold

    def test_futures_ticker_recognized_from_config(self):
        one_hour_ago = pd.Timestamp(datetime.now(pytz.utc) - timedelta(hours=1))
        ctx = _get_session_context("ES=F", one_hour_ago, 5700.0, 5690.0)
        self.assertEqual(ctx["instrument_type"], "futures")
        self.assertFalse(ctx["is_stale"])  # 1h < futures' 50h threshold

    def test_equity_survives_a_full_weekend_gap(self):
        # Friday close to Monday morning is ~63h — the old 27h threshold
        # flagged every equity as stale every Monday. See _get_session_context().
        sixty_three_hours_ago = pd.Timestamp(datetime.now(pytz.utc) - timedelta(hours=63))
        ctx = _get_session_context("SPY", sixty_three_hours_ago, 500.0, 495.0)
        self.assertEqual(ctx["instrument_type"], "equity")
        self.assertFalse(ctx["is_stale"])  # 63h < equity's 75h threshold

    def test_fx_survives_a_full_weekend_gap(self):
        forty_hours_ago = pd.Timestamp(datetime.now(pytz.utc) - timedelta(hours=40))
        ctx = _get_session_context("EURUSD=X", forty_hours_ago, 1.08, 1.07)
        self.assertEqual(ctx["instrument_type"], "fx")
        self.assertFalse(ctx["is_stale"])  # 40h < fx's 55h threshold

    def test_stale_session_label_matches_is_stale_verdict(self):
        # session_label used to append "STALE" purely off an age bucket
        # (>=72h), independent of the per-instrument threshold — so a
        # 63h-old equity bar (not stale under the 75h threshold) could
        # still say "STALE" in the label. It must track is_stale exactly.
        sixty_three_hours_ago = pd.Timestamp(datetime.now(pytz.utc) - timedelta(hours=63))
        ctx = _get_session_context("SPY", sixty_three_hours_ago, 500.0, 495.0)
        self.assertFalse(ctx["is_stale"])
        self.assertNotIn("STALE", ctx["session_label"])

    def test_intl_index_gets_local_session_note(self):
        four_hours_ago = pd.Timestamp(datetime.now(pytz.utc) - timedelta(hours=4))
        ctx = _get_session_context("^N225", four_hours_ago, 40000.0, 39500.0)
        self.assertEqual(ctx["instrument_type"], "intl_index")
        self.assertIn("local session", ctx["condition_note"])

    def test_naive_timestamp_treated_as_utc(self):
        naive = pd.Timestamp(datetime.now(pytz.utc).replace(tzinfo=None) - timedelta(hours=1))
        ctx = _get_session_context("SPY", naive, 500.0, 495.0)
        self.assertIsNotNone(ctx["data_age_hours"])
        self.assertLess(ctx["data_age_hours"], 2)

    def test_old_bar_is_stale_with_days_ago_label(self):
        four_days_ago = pd.Timestamp(datetime.now(pytz.utc) - timedelta(days=4))
        ctx = _get_session_context("SPY", four_days_ago, 500.0, 495.0)
        self.assertTrue(ctx["is_stale"])
        self.assertIn("d ago", ctx["session_label"])

    def test_daily_bar_uses_utc_date_not_et_shifted_date(self):
        # yfinance daily bars are stamped 00:00 UTC of the trading day.
        # Converting that to ET lands on 20:00 ET the PRIOR day — a Friday
        # bar (2026-09-04, a Friday) would read as Thursday. The trading
        # date must come from the UTC date directly for a midnight-UTC bar.
        friday_daily_bar = pd.Timestamp("2026-09-04T00:00:00", tz="UTC")
        ctx = _get_session_context("MSFT", friday_daily_bar, 420.0, 418.0)
        self.assertIn("Fri", ctx["condition_note"])
        self.assertNotIn("Thu", ctx["condition_note"])

    def test_daily_bar_has_no_data_lag_minutes(self):
        friday_daily_bar = pd.Timestamp("2026-09-04T00:00:00", tz="UTC")
        ctx = _get_session_context("MSFT", friday_daily_bar, 420.0, 418.0)
        self.assertIsNone(ctx["data_lag_minutes"])

    def test_intraday_bar_reports_data_lag_minutes(self):
        seven_minutes_ago = pd.Timestamp(datetime.now(pytz.utc) - timedelta(minutes=7))
        ctx = _get_session_context("MSFT", seven_minutes_ago, 420.0, 418.0)
        self.assertIsNotNone(ctx["data_lag_minutes"])
        self.assertAlmostEqual(ctx["data_lag_minutes"], 7, delta=0.5)

    def test_none_timestamp_has_none_exact_timestamp(self):
        ctx = _get_session_context("SPY", None, 500.0, 495.0)
        self.assertIsNone(ctx["exact_timestamp"])

    def test_daily_bar_exact_timestamp_shows_full_calendar_date(self):
        # Bug 1 (Thu/Fri off-by-one) fixed via is_daily_bar/trading_date —
        # exact_timestamp must carry that same fix, not the ET-shifted date.
        friday_daily_bar = pd.Timestamp("2026-09-04T00:00:00", tz="UTC")
        ctx = _get_session_context("MSFT", friday_daily_bar, 420.0, 418.0)
        self.assertEqual(ctx["exact_timestamp"], "Fri 2026-09-04 close")

    def test_intraday_bar_exact_timestamp_shows_et_datetime(self):
        bar = pd.Timestamp("2026-09-04T18:35:00", tz="UTC")  # 14:35 ET
        ctx = _get_session_context("MSFT", bar, 420.0, 418.0)
        self.assertEqual(ctx["exact_timestamp"], "2026-09-04 14:35 ET")


class TestGetPriorClose(unittest.TestCase):
    def test_normal_case_uses_second_to_last_bar(self):
        from equity.data.price_cache import _get_prior_close
        daily_close = pd.Series([100.0, 105.0, 110.0])  # iloc[-2] = 105.0
        self.assertEqual(_get_prior_close(daily_close, curr_price=112.0), 105.0)

    def test_too_short_series_falls_back_to_curr_price(self):
        from equity.data.price_cache import _get_prior_close
        daily_close = pd.Series([100.0])
        self.assertEqual(_get_prior_close(daily_close, curr_price=100.0), 100.0)

    def test_suspicious_match_steps_back_one_more_bar(self):
        # A holiday/thin-trading settlement bar can carry the prior
        # session's value forward, landing at iloc[-2] and matching
        # curr_price exactly — that's the false "0% change" this guards.
        from equity.data.price_cache import _get_prior_close
        daily_close = pd.Series([2650.0, 2700.0, 2700.0])  # iloc[-2] == curr_price
        self.assertEqual(_get_prior_close(daily_close, curr_price=2700.0), 2650.0)

    def test_suspicious_match_without_a_third_bar_returns_it_anyway(self):
        from equity.data.price_cache import _get_prior_close
        daily_close = pd.Series([2700.0, 2700.0])
        self.assertEqual(_get_prior_close(daily_close, curr_price=2700.0), 2700.0)


class TestExtendedIntradaySet(unittest.TestCase):
    def test_includes_crypto_and_futures(self):
        from equity.data.price_cache import _EXTENDED_INTRADAY_SET, _CRYPTO_TICKER_SET, _FUTURES_TICKER_SET
        self.assertTrue(_CRYPTO_TICKER_SET.issubset(_EXTENDED_INTRADAY_SET))
        self.assertTrue(_FUTURES_TICKER_SET.issubset(_EXTENDED_INTRADAY_SET))

    def test_excludes_plain_equities(self):
        from equity.data.price_cache import _EXTENDED_INTRADAY_SET
        self.assertNotIn("SPY", _EXTENDED_INTRADAY_SET)
        self.assertNotIn("MSFT", _EXTENDED_INTRADAY_SET)


class TestExtractClose(unittest.TestCase):
    def test_none_dataframe_returns_empty_series(self):
        from equity.data.price_cache import _extract_close
        result = _extract_close(None, "SPY")
        self.assertEqual(len(result), 0)

    def test_multiindex_missing_ticker_returns_empty_series(self):
        from equity.data.price_cache import _extract_close
        idx = pd.MultiIndex.from_tuples([("Close", "SPY")])
        df = pd.DataFrame([[500.0]], columns=idx)
        result = _extract_close(df, "QQQ")
        self.assertEqual(len(result), 0)

    def test_multiindex_present_ticker_returns_close_series(self):
        from equity.data.price_cache import _extract_close
        idx = pd.MultiIndex.from_tuples([("Close", "SPY")])
        df = pd.DataFrame([[500.0], [501.0]], columns=idx)
        result = _extract_close(df, "SPY")
        self.assertEqual(list(result), [500.0, 501.0])


if __name__ == "__main__":
    unittest.main()
