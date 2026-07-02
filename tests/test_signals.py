"""Tests for signals/sma_crossover.py."""

import unittest

from signals.sma_crossover import SignalResult, SMACrossoverSignal


class TestSMACrossoverSignal(unittest.TestCase):
    def setUp(self):
        self.signal = SMACrossoverSignal(period=20)

    def test_insufficient_data(self):
        prices = [100.0] * 19
        self.assertEqual(self.signal.compute(prices), 0)

    def test_bullish_signal(self):
        prices = [100.0] * 19 + [200.0]
        self.assertEqual(self.signal.compute(prices), 1)

    def test_bearish_signal(self):
        prices = [100.0] * 19 + [50.0]
        self.assertEqual(self.signal.compute(prices), -1)

    def test_neutral_signal(self):
        prices = [100.0] * 20
        self.assertEqual(self.signal.compute(prices), 0)

    def test_signal_result_metadata(self):
        prices = [100.0] * 19 + [200.0]
        result = self.signal.compute_with_metadata("BTC-USDC", prices)

        self.assertIsInstance(result, SignalResult)
        self.assertEqual(result.symbol, "BTC-USDC")
        self.assertEqual(result.signal_name, "sma_crossover")
        self.assertEqual(result.value, self.signal.compute(prices))

    def test_pure_no_side_effects(self):
        prices = [100.0] * 19 + [200.0]
        original = list(prices)

        first = self.signal.compute(prices)
        second = self.signal.compute(prices)

        self.assertEqual(first, second)
        self.assertEqual(prices, original)


if __name__ == "__main__":
    unittest.main()
