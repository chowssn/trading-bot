"""Tests for backtest/funding_rates.py."""

import os
import time
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import requests

from backtest.funding_rates import (
    CACHE_FILE,
    _classify_regime,
    _fetch_raw_funding_rates,
    _fetch_raw_funding_rates_bybit,
    fetch_funding_rates,
)


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if 400 <= self.status_code < 600:
            err = requests.exceptions.HTTPError(f"{self.status_code} error")
            err.response = self
            raise err

    def json(self):
        return self._payload


def _make_page(start_ms: int, count: int, rate: float = 0.0001):
    step_ms = 8 * 3600 * 1000
    return [
        {
            "symbol": "BTCUSDT",
            "fundingTime": start_ms + i * step_ms,
            "fundingRate": str(rate),
        }
        for i in range(count)
    ]


def _make_bybit_page(start_ms: int, count: int, cursor: str = "", rate: str = "0.0001"):
    step_ms = 8 * 3600 * 1000
    records = [
        {
            "symbol": "BTCUSDT",
            "fundingRateTimestamp": str(start_ms + i * step_ms),
            "fundingRate": rate,
        }
        for i in range(count)
    ]
    records.reverse()  # Bybit returns newest first
    return {"result": {"list": records, "nextPageCursor": cursor}, "retCode": 0}


class TestClassifyRegime(unittest.TestCase):
    def test_funding_regime_extreme_long(self):
        series = pd.Series([0.0006])
        result = _classify_regime(series)
        self.assertEqual(result["funding_regime"].iloc[0], "extreme_long")
        self.assertEqual(result["funding_adjustment"].iloc[0], 0.80)

    def test_funding_regime_extreme_short(self):
        series = pd.Series([-0.0003])
        result = _classify_regime(series)
        self.assertEqual(result["funding_regime"].iloc[0], "extreme_short")
        self.assertEqual(result["funding_adjustment"].iloc[0], 1.10)

    def test_funding_regime_normal(self):
        series = pd.Series([0.0002])
        result = _classify_regime(series)
        self.assertEqual(result["funding_regime"].iloc[0], "normal")
        self.assertEqual(result["funding_adjustment"].iloc[0], 1.00)

    def test_funding_adjustment_neutral_on_missing(self):
        series = pd.Series([np.nan])
        result = _classify_regime(series)
        self.assertTrue(pd.isna(result["funding_regime"].iloc[0]))
        self.assertEqual(result["funding_adjustment"].iloc[0], 1.0)


class TestPagination(unittest.TestCase):
    def test_pagination_covers_full_history(self):
        # Two 1000-record pages, back to back with no overlap. The oldest
        # record of page0 sits exactly at start_ms so pagination stops there.
        page1_start_ms = 1_700_000_000_000
        page1 = _make_page(page1_start_ms, 1000)

        step_ms = 8 * 3600 * 1000
        page0_start_ms = page1_start_ms - 1000 * step_ms
        page0 = _make_page(page0_start_ms, 1000)

        responses = [_FakeResponse(page1), _FakeResponse(page0)]
        start_time = pd.Timestamp(page0_start_ms, unit="ms", tz="UTC")
        end_time = pd.Timestamp(page1_start_ms + 999 * step_ms, unit="ms", tz="UTC")

        with mock.patch(
            "backtest.funding_rates.requests.get", side_effect=responses
        ) as mock_get:
            df = _fetch_raw_funding_rates("BTCUSDT", start_time, end_time)

        self.assertEqual(mock_get.call_count, 2)
        self.assertEqual(len(df), 2000)


class TestBybitPagination(unittest.TestCase):
    def test_bybit_pagination_covers_full_history(self):
        step_ms = 8 * 3600 * 1000
        page0_start_ms = 1_700_000_000_000
        page1_start_ms = page0_start_ms + 200 * step_ms

        page1 = _make_bybit_page(page1_start_ms, 200, cursor="next-cursor")
        page0 = _make_bybit_page(page0_start_ms, 200, cursor="")

        responses = [_FakeResponse(page1), _FakeResponse(page0)]
        start_time = pd.Timestamp(page0_start_ms, unit="ms", tz="UTC")

        with mock.patch(
            "backtest.funding_rates.requests.get", side_effect=responses
        ) as mock_get:
            df = _fetch_raw_funding_rates_bybit("BTCUSDT", start_time)

        self.assertEqual(mock_get.call_count, 2)
        self.assertEqual(len(df), 400)
        second_call_params = mock_get.call_args_list[1].kwargs["params"]
        self.assertEqual(second_call_params["cursor"], "next-cursor")


class TestBinanceToBybitFallback(unittest.TestCase):
    def test_falls_back_to_bybit_on_4xx(self):
        start_ms = 1_700_000_000_000
        step_ms = 8 * 3600 * 1000
        end_ms = start_ms + 9 * step_ms

        binance_error = _FakeResponse({}, status_code=451)
        bybit_page = _make_bybit_page(start_ms, 10, cursor="")

        with mock.patch(
            "backtest.funding_rates.requests.get",
            side_effect=[binance_error, _FakeResponse(bybit_page)],
        ) as mock_get:
            df = _fetch_raw_funding_rates(
                "BTCUSDT",
                pd.Timestamp(start_ms, unit="ms", tz="UTC"),
                pd.Timestamp(end_ms, unit="ms", tz="UTC"),
            )

        self.assertEqual(mock_get.call_count, 2)
        self.assertEqual(mock_get.call_args_list[1].args[0], "https://api.bybit.com/v5/market/funding/history")
        self.assertEqual(list(df.columns), ["fundingRate"])
        self.assertEqual(len(df), 10)

    def test_non_4xx_error_does_not_fall_back_to_bybit(self):
        with mock.patch(
            "backtest.funding_rates.requests.get",
            side_effect=ConnectionError("boom"),
        ) as mock_get:
            with self.assertRaises(ConnectionError):
                _fetch_raw_funding_rates(
                    "BTCUSDT",
                    pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1),
                    pd.Timestamp.now(tz="UTC"),
                )

        mock_get.assert_called_once()


class TestCacheBehavior(unittest.TestCase):
    def setUp(self):
        self._had_cache = CACHE_FILE.exists()
        if self._had_cache:
            self._backup = CACHE_FILE.read_bytes()
            self._backup_mtime = CACHE_FILE.stat().st_mtime

    def tearDown(self):
        if self._had_cache:
            CACHE_FILE.write_bytes(self._backup)
            os.utime(CACHE_FILE, (self._backup_mtime, self._backup_mtime))
        elif CACHE_FILE.exists():
            CACHE_FILE.unlink()

    def _write_cache(self, age_seconds: float):
        dates = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
        df = pd.DataFrame(
            {
                "fundingRate": [0.0001, 0.0001, 0.0001],
                "funding_rate_pct": [0.01, 0.01, 0.01],
                "funding_7d_avg": [0.0001, 0.0001, 0.0001],
                "funding_30d_avg": [0.0001, 0.0001, 0.0001],
                "funding_vs_30d_baseline": [0.0, 0.0, 0.0],
                "funding_regime": ["normal", "normal", "normal"],
                "funding_adjustment": [1.0, 1.0, 1.0],
            },
            index=dates,
        )
        df.index.name = "date"
        df.to_csv(CACHE_FILE)
        old_time = time.time() - age_seconds
        os.utime(CACHE_FILE, (old_time, old_time))

    def test_cache_ttl_8_hours(self):
        self._write_cache(age_seconds=60)  # fresh, well under 8h

        with mock.patch("backtest.funding_rates.requests.get") as mock_get:
            result = fetch_funding_rates()

        mock_get.assert_not_called()
        self.assertEqual(len(result), 3)

    def test_graceful_fallback_on_error(self):
        self._write_cache(age_seconds=9 * 60 * 60)  # stale, forces a refetch attempt

        with mock.patch(
            "backtest.funding_rates.requests.get", side_effect=ConnectionError("boom")
        ):
            result = fetch_funding_rates()

        self.assertEqual(len(result), 3)
        self.assertEqual(result["funding_regime"].iloc[0], "normal")


if __name__ == "__main__":
    unittest.main()
