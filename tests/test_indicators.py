"""Tests for backtest/data_fetcher.py's streak-counting helper."""

import unittest

import pandas as pd

from backtest.data_fetcher import _consecutive_streak


class TestConsecutiveStreak(unittest.TestCase):
    def test_streak_resumes_correctly_after_single_break(self):
        cond = pd.Series([True, True, False, True, True])
        streak = _consecutive_streak(cond)
        self.assertEqual(list(streak), [1, 2, 0, 1, 2])

    def test_all_true(self):
        cond = pd.Series([True, True, True])
        streak = _consecutive_streak(cond)
        self.assertEqual(list(streak), [1, 2, 3])

    def test_all_false(self):
        cond = pd.Series([False, False, False])
        streak = _consecutive_streak(cond)
        self.assertEqual(list(streak), [0, 0, 0])

    def test_nan_treated_as_false(self):
        cond = pd.Series([True, None, True, True])
        streak = _consecutive_streak(cond)
        self.assertEqual(list(streak), [1, 0, 1, 2])

    def test_alternating(self):
        cond = pd.Series([True, False, True, False, True])
        streak = _consecutive_streak(cond)
        self.assertEqual(list(streak), [1, 0, 1, 0, 1])


if __name__ == "__main__":
    unittest.main()
