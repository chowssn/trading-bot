"""Tests for backtest/etf_flows.py and its integration into backtest/data_fetcher.py."""

import unittest
from io import StringIO
from unittest import mock

import numpy as np
import pandas as pd

from backtest.data_fetcher import _merge_etf_flows
from backtest.etf_flows import (
    FLOW_COLUMNS,
    _compute_derived_metrics,
    _consecutive_streak,
    _load_sosovalue_seed,
    _merge_seed_and_extension,
)


def _flow_frame(daily_flow_musd: pd.Series, **extra_columns) -> pd.DataFrame:
    """Build a merged-source frame (daily_flow_musd + per-ETF columns) for _compute_derived_metrics."""
    cols = {"daily_flow_musd": daily_flow_musd}
    for name in ("ibit_flow_musd", "fbtc_flow_musd", "arkb_flow_musd", "bitb_flow_musd", "gbtc_flow_musd"):
        cols[name] = extra_columns.get(name, pd.Series(np.nan, index=daily_flow_musd.index))
    return pd.DataFrame(cols, index=daily_flow_musd.index)


class TestSoSoValueSeedParsing(unittest.TestCase):
    CSV = (
        "Date\tDaily Total BTC Inflow(USD)\tCumulative Total Net Inflow(USD)\t"
        "Total Value Traded(USD)\tTotal Net Assets(USD)\n"
        "1/12/2024\t190,973,808.60\t819,043,388.60\t3,185,166,205\t27,932,766,328.90\n"
        "1/11/2024\t628,069,580\t628,069,580\t4,661,432,563\t29,375,231,230\n"
        "1/10/2024\t-80,581,829.04\tNot updated\t1,926,440,576\t28,710,277,596.15\n"
    )

    def test_parses_dates_converts_to_millions_and_sorts_ascending(self):
        seed = _load_sosovalue_seed(StringIO(self.CSV))

        self.assertEqual(list(seed.index), sorted(seed.index))
        self.assertEqual(
            list(seed.index),
            [
                pd.Timestamp("2024-01-10", tz="UTC"),
                pd.Timestamp("2024-01-11", tz="UTC"),
                pd.Timestamp("2024-01-12", tz="UTC"),
            ],
        )
        self.assertAlmostEqual(seed.loc[pd.Timestamp("2024-01-10", tz="UTC"), "daily_flow_musd"], -80.58182904)
        self.assertAlmostEqual(seed.loc[pd.Timestamp("2024-01-11", tz="UTC"), "daily_flow_musd"], 628.069580)
        self.assertAlmostEqual(seed.loc[pd.Timestamp("2024-01-12", tz="UTC"), "daily_flow_musd"], 190.97380860)


class TestMergeSeedAndExtension(unittest.TestCase):
    def test_extension_appended_and_sosovalue_preferred_on_overlap(self):
        seed_dates = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
        seed = pd.DataFrame({"daily_flow_musd": [100.0, 200.0, 300.0]}, index=seed_dates)

        # Extension covers one overlapping day (should be ignored) plus two new days.
        ext_dates = pd.date_range("2024-01-03", periods=3, freq="D", tz="UTC")
        extension = pd.DataFrame(
            {
                "daily_flow_musd": [999.0, 400.0, 500.0],
                "ibit_flow_musd": [10.0, 20.0, 30.0],
                "fbtc_flow_musd": [1.0, 2.0, 3.0],
                "arkb_flow_musd": [0.1, 0.2, 0.3],
                "bitb_flow_musd": [0.01, 0.02, 0.03],
                "gbtc_flow_musd": [-1.0, -2.0, -3.0],
            },
            index=ext_dates,
        )

        merged = _merge_seed_and_extension(seed, extension)

        # SoSoValue wins on the overlapping date (2024-01-03): 300.0, not 999.0.
        self.assertEqual(merged.loc[pd.Timestamp("2024-01-03", tz="UTC"), "daily_flow_musd"], 300.0)
        # Seed-only dates have no per-ETF breakdown.
        self.assertTrue(pd.isna(merged.loc[pd.Timestamp("2024-01-01", tz="UTC"), "ibit_flow_musd"]))
        # Extension-only dates are appended with their per-ETF values intact.
        self.assertEqual(merged.loc[pd.Timestamp("2024-01-04", tz="UTC"), "daily_flow_musd"], 400.0)
        self.assertEqual(merged.loc[pd.Timestamp("2024-01-04", tz="UTC"), "ibit_flow_musd"], 20.0)
        self.assertEqual(merged.loc[pd.Timestamp("2024-01-05", tz="UTC"), "daily_flow_musd"], 500.0)

    def test_gaps_forward_filled_up_to_three_days(self):
        seed = pd.DataFrame(
            {"daily_flow_musd": [100.0]},
            index=pd.DatetimeIndex([pd.Timestamp("2024-01-01", tz="UTC")]),
        )
        extension = pd.DataFrame(
            {
                "daily_flow_musd": [200.0],
                "ibit_flow_musd": [1.0],
                "fbtc_flow_musd": [1.0],
                "arkb_flow_musd": [1.0],
                "bitb_flow_musd": [1.0],
                "gbtc_flow_musd": [1.0],
            },
            index=pd.DatetimeIndex([pd.Timestamp("2024-01-05", tz="UTC")]),
        )

        merged = _merge_seed_and_extension(seed, extension)

        # 2024-01-02, 01-03, 01-04 are gaps (weekend/holiday) filled from 01-01's value.
        for day in ("2024-01-02", "2024-01-03", "2024-01-04"):
            self.assertEqual(merged.loc[pd.Timestamp(day, tz="UTC"), "daily_flow_musd"], 100.0)
        self.assertEqual(merged.loc[pd.Timestamp("2024-01-05", tz="UTC"), "daily_flow_musd"], 200.0)


class TestComputeDerivedMetrics(unittest.TestCase):
    def test_flow_score_buckets(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
        # constant daily flow => flow_5d_sum on day 5 == daily_flow_musd * 5
        cases = {
            600: 1.0,   # 5d sum = 3000, > 500
            50: 0.75,   # 5d sum = 250, in (100, 500]
            10: 0.5,    # 5d sum = 50, within +-100
            -50: 0.25,  # 5d sum = -250, in [-500, -100)
            -600: 0.0,  # 5d sum = -3000, < -500
        }
        for daily, expected_score in cases.items():
            flows = pd.Series([daily] * 5, index=dates)
            df = _compute_derived_metrics(_flow_frame(flows))
            self.assertEqual(df["flow_score"].iloc[-1], expected_score)

    def test_flow_regime_thresholds(self):
        dates = pd.date_range("2024-01-01", periods=20, freq="D", tz="UTC")

        accumulation = pd.Series([200] * 20, index=dates)
        self.assertEqual(
            _compute_derived_metrics(_flow_frame(accumulation))["flow_regime"].iloc[-1], "accumulation"
        )

        distribution = pd.Series([-200] * 20, index=dates)
        self.assertEqual(
            _compute_derived_metrics(_flow_frame(distribution))["flow_regime"].iloc[-1], "distribution"
        )

        neutral = pd.Series([10] * 20, index=dates)
        self.assertEqual(_compute_derived_metrics(_flow_frame(neutral))["flow_regime"].iloc[-1], "neutral")

    def test_consecutive_negative_resets_on_positive(self):
        dates = pd.date_range("2024-01-01", periods=6, freq="D", tz="UTC")
        flows = pd.Series([-10, -10, -10, 5, -10, -10], index=dates)
        df = _compute_derived_metrics(_flow_frame(flows))
        self.assertEqual(list(df["flow_consecutive_negative"]), [1, 2, 3, 0, 1, 2])

    def test_exit_accelerator_requires_streak_and_5d_sum(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")

        # 5 consecutive negative days, 5d sum well below -1000 => accelerator fires
        strong_negative = pd.Series([-300] * 5, index=dates)
        df = _compute_derived_metrics(_flow_frame(strong_negative))
        self.assertTrue(df["flow_exit_accelerator"].iloc[-1])

        # 5 consecutive negative days but 5d sum shallow => no accelerator
        mild_negative = pd.Series([-100] * 5, index=dates)
        df = _compute_derived_metrics(_flow_frame(mild_negative))
        self.assertFalse(df["flow_exit_accelerator"].iloc[-1])

    def test_flow_long_trend_compares_20d_pace_to_sma200(self):
        # 200 days of small positive flow to establish flow_sma200, then a
        # strong final 20-day burst that should push 20d pace above SMA200*20.
        base_dates = pd.date_range("2024-01-01", periods=200, freq="D", tz="UTC")
        burst_dates = pd.date_range(base_dates[-1] + pd.Timedelta(days=1), periods=20, freq="D", tz="UTC")
        flows = pd.Series([10] * 200 + [500] * 20, index=base_dates.append(burst_dates))
        df = _compute_derived_metrics(_flow_frame(flows))
        self.assertEqual(df["flow_long_trend"].iloc[-1], 1.0)

    def test_ex_gbtc_flow_musd_excludes_gbtc(self):
        dates = pd.date_range("2024-01-01", periods=1, freq="D", tz="UTC")
        flows = pd.Series([0.0], index=dates)
        df = _compute_derived_metrics(
            _flow_frame(
                flows,
                ibit_flow_musd=pd.Series([100.0], index=dates),
                fbtc_flow_musd=pd.Series([50.0], index=dates),
                arkb_flow_musd=pd.Series([10.0], index=dates),
                bitb_flow_musd=pd.Series([5.0], index=dates),
                gbtc_flow_musd=pd.Series([-200.0], index=dates),
            )
        )
        self.assertEqual(df["ex_gbtc_flow_musd"].iloc[0], 165.0)

    def test_ex_gbtc_flow_score_falls_back_to_flow_score_when_unavailable(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
        flows = pd.Series([600] * 5, index=dates)  # flow_score == 1.0
        df = _compute_derived_metrics(_flow_frame(flows))  # per-ETF cols all NaN
        self.assertEqual(df["ex_gbtc_flow_score"].iloc[-1], df["flow_score"].iloc[-1])

    def test_gbtc_outflow_pressure_requires_three_consecutive_days(self):
        dates = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
        flows = pd.Series([0.0] * 4, index=dates)
        gbtc = pd.Series([-150.0, -150.0, -50.0, -150.0], index=dates)
        df = _compute_derived_metrics(_flow_frame(flows, gbtc_flow_musd=gbtc))
        self.assertEqual(list(df["gbtc_outflow_pressure"]), [False, False, False, False])

        gbtc_three_in_a_row = pd.Series([-150.0, -150.0, -150.0, -150.0], index=dates)
        df = _compute_derived_metrics(_flow_frame(flows, gbtc_flow_musd=gbtc_three_in_a_row))
        self.assertEqual(list(df["gbtc_outflow_pressure"]), [False, False, True, True])


class TestConsecutiveStreak(unittest.TestCase):
    def test_streak_resumes_correctly_after_single_break(self):
        cond = pd.Series([True, True, False, True, True])
        self.assertEqual(list(_consecutive_streak(cond)), [1, 2, 0, 1, 2])

    def test_nan_treated_as_false(self):
        cond = pd.Series([True, None, True, True])
        self.assertEqual(list(_consecutive_streak(cond)), [1, 0, 1, 2])


class TestNeutralFallback(unittest.TestCase):
    def test_flow_score_defaults_to_neutral_when_flow_data_unavailable(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
        macro_df = pd.DataFrame({"spy_close": [1.0] * 5}, index=dates)

        empty_flows = pd.DataFrame(columns=FLOW_COLUMNS)
        empty_flows.index.name = "date"

        with mock.patch("backtest.data_fetcher.fetch_etf_flows", return_value=empty_flows):
            merged = _merge_etf_flows(macro_df, days_back=1460)

        self.assertTrue((merged["flow_score"] == 0.5).all())
        self.assertFalse(merged["flow_score"].isna().any())

    def test_flow_score_neutral_beyond_the_3day_ffill_gap(self):
        # A gap of 3 days or less is bridged by ffill; beyond that, flow_score
        # should fall back to 0.5 rather than staying NaN.
        dates = pd.date_range("2024-01-01", periods=6, freq="D", tz="UTC")
        macro_df = pd.DataFrame({"spy_close": [1.0] * 6}, index=dates)

        partial_flows = pd.DataFrame(
            {"flow_score": [1.0, np.nan, np.nan, np.nan, np.nan, np.nan]},
            index=dates,
        )
        partial_flows.index.name = "date"

        with mock.patch("backtest.data_fetcher.fetch_etf_flows", return_value=partial_flows):
            merged = _merge_etf_flows(macro_df, days_back=1460)

        # Days 1-3 (within the 3-day ffill limit) carry the last known value forward.
        self.assertEqual(list(merged["flow_score"].iloc[0:4]), [1.0, 1.0, 1.0, 1.0])
        # Days 4-5 exceed the ffill limit and fall back to neutral instead of NaN.
        self.assertEqual(list(merged["flow_score"].iloc[4:6]), [0.5, 0.5])
        self.assertFalse(merged["flow_score"].isna().any())


if __name__ == "__main__":
    unittest.main()
