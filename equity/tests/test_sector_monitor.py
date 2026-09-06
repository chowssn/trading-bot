"""Tests for equity/brief/sector_monitor.py's options classification and correlation formatting."""

import unittest

import pandas as pd

from equity.brief import sector_monitor


class TestClassifyOptions(unittest.TestCase):
    def test_no_data(self):
        self.assertEqual(sector_monitor.classify_options(None, None), "NO_DATA")

    def test_normal_below_all_thresholds(self):
        self.assertEqual(sector_monitor.classify_options(30, 1.0), "NORMAL")

    def test_iv_elevated(self):
        self.assertEqual(sector_monitor.classify_options(65, None), "IV_ELEVATED")

    def test_iv_extreme(self):
        self.assertEqual(sector_monitor.classify_options(110, None), "IV_EXTREME")

    def test_put_heavy(self):
        self.assertEqual(sector_monitor.classify_options(None, 3.0), "PUT_HEAVY")

    def test_extreme_put(self):
        self.assertEqual(sector_monitor.classify_options(None, 5.0), "EXTREME_PUT")

    def test_call_heavy(self):
        self.assertEqual(sector_monitor.classify_options(None, 0.2), "CALL_HEAVY")

    def test_combined_signal(self):
        self.assertEqual(sector_monitor.classify_options(65, 3.0), "IV_ELEVATED + PUT_HEAVY")

    def test_boundary_values_are_exclusive(self):
        # Thresholds are strict ">"/"<" — a value exactly at the line is still NORMAL.
        self.assertEqual(sector_monitor.classify_options(60, 2.5), "NORMAL")


class TestClearsLiquidityGates(unittest.TestCase):
    def test_thin_chain_fails_gate(self):
        # Below MIN_TOTAL_VOLUME entirely.
        self.assertFalse(sector_monitor._clears_liquidity_gates(total_volume=100, total_oi=10))

    def test_quoted_not_traded_fails_volume_oi_gate(self):
        # High volume, but not enough relative to open interest.
        self.assertFalse(sector_monitor._clears_liquidity_gates(total_volume=6000, total_oi=10000))

    def test_liquid_actively_traded_chain_passes(self):
        self.assertTrue(sector_monitor._clears_liquidity_gates(total_volume=10000, total_oi=1000))


class TestFormatCorrelationClusters(unittest.TestCase):
    def _corr(self, values: dict) -> pd.DataFrame:
        tickers = sorted({t for pair in values for t in pair})
        df = pd.DataFrame(1.0, index=tickers, columns=tickers)
        for (a, b), corr in values.items():
            df.loc[a, b] = corr
            df.loc[b, a] = corr
        return df

    def test_no_clusters_below_three_members(self):
        corr = self._corr({("A", "B"): 0.9})
        self.assertIn("No significant concentration clusters", sector_monitor._format_correlation_clusters(corr))

    def test_three_member_cluster_detected(self):
        corr = self._corr({
            ("A", "B"): 0.85, ("A", "C"): 0.80, ("B", "C"): 0.75,
        })
        result = sector_monitor._format_correlation_clusters(corr)
        self.assertIn("A, B, C", result)

    def test_high_vs_moderate_risk_label(self):
        corr = self._corr({
            ("A", "B"): 0.90, ("A", "C"): 0.90, ("B", "C"): 0.90,
        })
        result = sector_monitor._format_correlation_clusters(corr)
        self.assertIn("[HIGH]", result)


class TestFormatTopHedges(unittest.TestCase):
    def test_only_negative_below_threshold_shown(self):
        tickers = ["A", "B", "C"]
        df = pd.DataFrame(1.0, index=tickers, columns=tickers)
        df.loc["A", "B"] = df.loc["B", "A"] = -0.5
        df.loc["A", "C"] = df.loc["C", "A"] = -0.1  # above CORRELATION_HEDGE_THRESHOLD, not a hedge
        result = sector_monitor._format_top_hedges(df)
        self.assertIn("A ↔ B", result)
        self.assertNotIn("A ↔ C", result)

    def test_no_hedges_detected(self):
        tickers = ["A", "B"]
        df = pd.DataFrame(1.0, index=tickers, columns=tickers)
        df.loc["A", "B"] = df.loc["B", "A"] = 0.5
        self.assertIn("No significant natural hedges", sector_monitor._format_top_hedges(df))


class TestFormatOptionsFlags(unittest.TestCase):
    def test_normal_and_no_data_excluded(self):
        options_data = {
            "AAA": {"iv30": 30, "put_call_ratio": 1.0, "signal": "NORMAL"},
            "BBB": {"iv30": None, "put_call_ratio": None, "signal": "NO_DATA"},
            "CCC": {"iv30": 65, "put_call_ratio": None, "signal": "IV_ELEVATED"},
        }
        result = sector_monitor._format_options_flags(options_data)
        self.assertNotIn("AAA", result)
        self.assertNotIn("BBB", result)
        self.assertIn("CCC", result)

    def test_all_normal_gives_clean_message(self):
        options_data = {"AAA": {"iv30": 30, "put_call_ratio": 1.0, "signal": "NORMAL"}}
        result = sector_monitor._format_options_flags(options_data)
        self.assertIn("No unusual positioning", result)


if __name__ == "__main__":
    unittest.main()
