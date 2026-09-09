"""Tests for equity/brief/performance_tracker.py's CAGR fallback (the "5Y
return N/A" fix): a ticker without a full 3Y/5Y of history gets its
longest-available annualized (or plain, if under 1Y) return labeled by
actual span, rather than a bare None a caller renders as "n/a".
"""

import unittest

import pandas as pd

from equity.brief.performance_tracker import _cagr_with_fallback, _fmt_cagr_cell


def _close_series(n_days: int, start: float, end: float) -> pd.Series:
    """A business-day close series of `n_days` points, linearly
    interpolated from `start` to `end` (monotonic growth is all
    _cagr_with_fallback needs). Business-day frequency (~252/year) matches
    both real yfinance daily data and _cagr_with_fallback's own
    actual_days/252 (years) and /21 (months) trading-day assumptions —
    calendar-day frequency would make `n_days` mean something different
    to _price_years_ago()'s calendar-day lookback than to those.
    """
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    values = [start + (end - start) * i / max(n_days - 1, 1) for i in range(n_days)]
    return pd.Series(values, index=dates)


class TestCagrWithFallback(unittest.TestCase):
    def test_full_history_uses_nominal_label(self):
        close = _close_series(252 * 6, 100, 200)  # 6 years — plenty for 5Y
        pct, label = _cagr_with_fallback(close, 5)
        self.assertIsNotNone(pct)
        self.assertEqual(label, "5Y")

    def test_short_history_falls_back_to_actual_span_annualized(self):
        # ~18 months of history — not enough for a 5Y CAGR.
        close = _close_series(int(252 * 1.5), 100, 130)
        pct, label = _cagr_with_fallback(close, 5)
        self.assertIsNotNone(pct)
        self.assertIn("M ann", label)
        self.assertNotEqual(label, "5Y")

    def test_very_short_history_falls_back_to_plain_return(self):
        # ~8 months — under a year, so no annualization.
        close = _close_series(21 * 8, 100, 108)
        pct, label = _cagr_with_fallback(close, 3)
        self.assertIsNotNone(pct)
        self.assertNotIn("ann", label)
        self.assertIn("M", label)

    def test_almost_no_history_returns_none_with_na_label(self):
        close = _close_series(3, 100, 101)
        pct, label = _cagr_with_fallback(close, 5)
        self.assertIsNone(pct)
        self.assertIn("N/A", label)

    def test_never_raises_on_zero_first_price(self):
        close = _close_series(21 * 8, 0, 108)
        pct, label = _cagr_with_fallback(close, 3)
        self.assertIsNone(pct)
        self.assertIn("N/A", label)


class TestFmtCagrCell(unittest.TestCase):
    def test_none_renders_as_na(self):
        self.assertEqual(_fmt_cagr_cell(None, "5Y N/A", "5Y"), "n/a")

    def test_nominal_label_renders_plain(self):
        self.assertEqual(_fmt_cagr_cell(9.2, "5Y", "5Y"), "+9.2%")

    def test_fallback_label_is_annotated(self):
        self.assertEqual(_fmt_cagr_cell(9.2, "18M ann", "5Y"), "+9.2% (18M ann)")


if __name__ == "__main__":
    unittest.main()
