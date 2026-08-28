"""Tests for equity/screener/quality_scorer.py and equity/screener/timing_signal.py."""

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from equity.screener import quality_scorer, timing_signal

QUARTERLY_DATES = pd.to_datetime(["2026-06-30", "2026-03-31", "2025-12-31", "2025-09-30"])
ANNUAL_DATES = pd.to_datetime(["2025-09-30", "2024-09-30", "2023-09-30", "2022-09-30"])


def _quarterly_cashflow(cfo_q: float, da_q: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Operating Cash Flow": [cfo_q] * 4,
            "Depreciation And Amortization": [da_q] * 4,
        },
        index=QUARTERLY_DATES,
    ).T


def _quarterly_income_stmt(net_income_q: float, op_income_q: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Net Income": [net_income_q] * 4,
            "Operating Income": [op_income_q] * 4,
        },
        index=QUARTERLY_DATES,
    ).T


def _quarterly_balance_sheet(net_debt: float) -> pd.DataFrame:
    return pd.DataFrame({"Net Debt": [net_debt] * 4}, index=QUARTERLY_DATES).T


def _annual_income_stmt(revenue: list, op_income: list, shares: list) -> pd.DataFrame:
    return pd.DataFrame(
        {"Total Revenue": revenue, "Operating Income": op_income, "Basic Average Shares": shares},
        index=ANNUAL_DATES,
    ).T


def _annual_cashflow(da: list) -> pd.DataFrame:
    return pd.DataFrame({"Depreciation And Amortization": da}, index=ANNUAL_DATES).T


def _build_mock_ticker(
    cfo_q, net_income_q, op_income_q, da_q, net_debt,
    revenue, annual_op_income, annual_da, shares,
    forward_pe, trailing_pe, sector,
) -> MagicMock:
    tk = MagicMock()
    tk.quarterly_cashflow = _quarterly_cashflow(cfo_q, da_q)
    tk.quarterly_income_stmt = _quarterly_income_stmt(net_income_q, op_income_q)
    tk.quarterly_balance_sheet = _quarterly_balance_sheet(net_debt)
    tk.income_stmt = _annual_income_stmt(revenue, annual_op_income, shares)
    tk.cashflow = _annual_cashflow(annual_da)
    tk.info = {"forwardPE": forward_pe, "trailingPE": trailing_pe, "sector": sector}
    return tk


class TestQualityScoreTier1(unittest.TestCase):
    @patch("equity.screener.quality_scorer._save_cache")
    @patch("equity.screener.quality_scorer._load_cached", return_value=None)
    @patch("equity.screener.quality_scorer.calculate_roic")
    @patch("equity.screener.quality_scorer.yf.Ticker")
    def test_quality_score_tier1(self, mock_ticker_cls, mock_calculate_roic, _mock_load, _mock_save):
        # CFO ($40M/q) > Net income ($30M/q); net debt ($100M) / EBITDA (~$280M T12M) ~= 0.36x;
        # shares shrinking (buyback); revenue growing ~60% over 3y (>15% CAGR); 30% EBITDA margin;
        # forward PE < trailing PE (improving).
        mock_ticker_cls.return_value = _build_mock_ticker(
            cfo_q=40_000_000, net_income_q=30_000_000, op_income_q=60_000_000, da_q=10_000_000,
            net_debt=100_000_000,
            revenue=[160_000_000, 140_000_000, 120_000_000, 100_000_000],
            annual_op_income=[38_400_000, 33_600_000, 28_800_000, 24_000_000],
            annual_da=[9_600_000, 8_400_000, 7_200_000, 6_000_000],
            shares=[95_000_000, 97_000_000, 99_000_000, 100_000_000],
            forward_pe=15.0, trailing_pe=20.0, sector="Industrials",
        )
        mock_calculate_roic.return_value = {"roic_current": 30.0, "roic_5y_avg": 28.0}

        result = quality_scorer.score_ticker("TIER1", api_key="fake-key")

        self.assertGreaterEqual(result["quality_score"], 70)
        self.assertEqual(result["tier"], "tier1")
        self.assertEqual(result["red_flags"], [])
        self.assertEqual(result["share_count_direction"], "buyback")
        self.assertTrue(result["cfo_gte_ni"])


class TestQualityScoreLowRoic(unittest.TestCase):
    @patch("equity.screener.quality_scorer._save_cache")
    @patch("equity.screener.quality_scorer._load_cached", return_value=None)
    @patch("equity.screener.quality_scorer.calculate_roic")
    @patch("equity.screener.quality_scorer.yf.Ticker")
    def test_quality_score_low_roic(self, mock_ticker_cls, mock_calculate_roic, _mock_load, _mock_save):
        # Weak fundamentals across the board on top of low ROIC, so score stays well under 30.
        mock_ticker_cls.return_value = _build_mock_ticker(
            cfo_q=5_000_000, net_income_q=10_000_000, op_income_q=8_000_000, da_q=2_000_000,
            net_debt=500_000_000,
            revenue=[100_000_000, 102_000_000, 104_000_000, 106_000_000],
            annual_op_income=[8_000_000, 8_200_000, 8_400_000, 8_600_000],
            annual_da=[2_000_000, 2_000_000, 2_000_000, 2_000_000],
            shares=[110_000_000, 105_000_000, 102_000_000, 100_000_000],
            forward_pe=25.0, trailing_pe=20.0, sector="Energy",
        )
        mock_calculate_roic.return_value = {"roic_current": 8.0, "roic_5y_avg": 7.5}

        result = quality_scorer.score_ticker("LOWROIC", api_key="fake-key")

        self.assertIn("low_roic", result["red_flags"])
        self.assertLess(result["quality_score"], 30)


class TestQualityScoreErrorHandling(unittest.TestCase):
    @patch("equity.screener.quality_scorer._load_cached", return_value=None)
    @patch("equity.screener.quality_scorer.yf.Ticker", side_effect=RuntimeError("boom"))
    def test_unhandled_exception_never_raises(self, _mock_ticker_cls, _mock_load):
        result = quality_scorer.score_ticker("BROKEN", api_key="fake-key")

        self.assertEqual(result["quality_score"], 0)
        self.assertEqual(result["tier"], "error")
        self.assertTrue(any("boom" in w for w in result["data_warnings"]))


class TestTimingSignal(unittest.TestCase):
    def test_timing_entry(self):
        result = timing_signal.classify_timing(rsi_14d=35, rsi_14d_direction="rising")
        self.assertEqual(result["timing_signal"], "ENTRY")

    def test_timing_wait(self):
        result = timing_signal.classify_timing(rsi_14d=20, rsi_14d_direction="falling")
        self.assertEqual(result["timing_signal"], "WAIT")

    def test_timing_watch(self):
        result = timing_signal.classify_timing(rsi_14d=38, rsi_14d_direction="neutral")
        self.assertEqual(result["timing_signal"], "WATCH")

    def test_timing_wait_high_rsi(self):
        result = timing_signal.classify_timing(rsi_14d=65, rsi_14d_direction="rising")
        self.assertEqual(result["timing_signal"], "WAIT")

    def test_timing_watch_approaching_from_above(self):
        result = timing_signal.classify_timing(rsi_14d=45, rsi_14d_direction="rising")
        self.assertEqual(result["timing_signal"], "WATCH")


if __name__ == "__main__":
    unittest.main()
