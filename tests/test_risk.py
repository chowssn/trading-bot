"""Tests for risk/risk_manager.py."""

import os
import shutil
import tempfile
import unittest

from risk.risk_manager import RiskManager


class TestRiskManager(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.kill_switch_path = os.path.join(self.tmp_dir, "KILL_SWITCH")
        self.risk = RiskManager(
            max_order_size_usd=10.0,
            max_exposure_usd=50.0,
            min_order_size_usd=1.0,
            max_orders_per_hour=5,
            kill_switch_path=self.kill_switch_path,
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_approved(self):
        approved, reason = self.risk.check(
            signal_value=1,
            order_size_usd=5.0,
            current_exposure_usd=0.0,
            paper_mode=True,
        )
        self.assertTrue(approved)
        self.assertEqual(reason, "approved")

    def test_kill_switch_blocks(self):
        with open(self.kill_switch_path, "w") as f:
            f.write("")

        approved, reason = self.risk.check(
            signal_value=1,
            order_size_usd=5.0,
            current_exposure_usd=0.0,
            paper_mode=True,
        )
        self.assertFalse(approved)
        self.assertIn("kill switch", reason)

    def test_kill_switch_clears(self):
        with open(self.kill_switch_path, "w") as f:
            f.write("")
        os.remove(self.kill_switch_path)

        approved, reason = self.risk.check(
            signal_value=1,
            order_size_usd=5.0,
            current_exposure_usd=0.0,
            paper_mode=True,
        )
        self.assertTrue(approved)
        self.assertEqual(reason, "approved")

    def test_paper_mode_required(self):
        approved, reason = self.risk.check(
            signal_value=1,
            order_size_usd=5.0,
            current_exposure_usd=0.0,
            paper_mode=False,
        )
        self.assertFalse(approved)
        self.assertEqual(reason, "paper mode required")

    def test_no_signal_blocked(self):
        approved, reason = self.risk.check(
            signal_value=0,
            order_size_usd=5.0,
            current_exposure_usd=0.0,
            paper_mode=True,
        )
        self.assertFalse(approved)
        self.assertEqual(reason, "no signal")

    def test_below_minimum_size(self):
        approved, reason = self.risk.check(
            signal_value=1,
            order_size_usd=0.50,
            current_exposure_usd=0.0,
            paper_mode=True,
        )
        self.assertFalse(approved)
        self.assertEqual(reason, "order below minimum size")

    def test_exceeds_max_size(self):
        approved, reason = self.risk.check(
            signal_value=1,
            order_size_usd=15.0,
            current_exposure_usd=0.0,
            paper_mode=True,
        )
        self.assertFalse(approved)
        self.assertEqual(reason, "order exceeds max size")

    def test_exceeds_max_exposure(self):
        approved, reason = self.risk.check(
            signal_value=1,
            order_size_usd=10.0,
            current_exposure_usd=45.0,
            paper_mode=True,
        )
        self.assertFalse(approved)
        self.assertEqual(reason, "exceeds max exposure")

    def test_rate_limit(self):
        for _ in range(5):
            approved, reason = self.risk.check(
                signal_value=1,
                order_size_usd=5.0,
                current_exposure_usd=0.0,
                paper_mode=True,
            )
            self.assertTrue(approved)
            self.assertEqual(reason, "approved")

        approved, reason = self.risk.check(
            signal_value=1,
            order_size_usd=5.0,
            current_exposure_usd=0.0,
            paper_mode=True,
        )
        self.assertFalse(approved)
        self.assertEqual(reason, "order rate limit reached")

    def test_reset_clears_rate_limit(self):
        for _ in range(5):
            self.risk.check(
                signal_value=1,
                order_size_usd=5.0,
                current_exposure_usd=0.0,
                paper_mode=True,
            )

        self.risk.reset_order_history()

        approved, reason = self.risk.check(
            signal_value=1,
            order_size_usd=5.0,
            current_exposure_usd=0.0,
            paper_mode=True,
        )
        self.assertTrue(approved)
        self.assertEqual(reason, "approved")

    def test_drawdown_within_threshold_approved(self):
        approved, reason = self.risk.check_portfolio_drawdown(
            current_value=90.0,
            peak_value=100.0,
            max_drawdown_threshold=0.15,
        )
        self.assertTrue(approved)
        self.assertEqual(reason, "approved")

    def test_drawdown_exceeds_threshold_blocked(self):
        approved, reason = self.risk.check_portfolio_drawdown(
            current_value=80.0,
            peak_value=100.0,
            max_drawdown_threshold=0.15,
        )
        self.assertFalse(approved)
        self.assertEqual(
            reason, "portfolio drawdown limit reached — manual reset required"
        )

    def test_drawdown_circuit_stays_tripped_until_reset(self):
        self.risk.check_portfolio_drawdown(
            current_value=80.0,
            peak_value=100.0,
            max_drawdown_threshold=0.15,
        )

        approved, reason = self.risk.check_portfolio_drawdown(
            current_value=100.0,
            peak_value=100.0,
            max_drawdown_threshold=0.15,
        )
        self.assertFalse(approved)
        self.assertEqual(
            reason, "portfolio drawdown limit reached — manual reset required"
        )

        self.risk.reset_drawdown_circuit()

        approved, reason = self.risk.check_portfolio_drawdown(
            current_value=100.0,
            peak_value=100.0,
            max_drawdown_threshold=0.15,
        )
        self.assertTrue(approved)
        self.assertEqual(reason, "approved")

    def test_drawdown_peak_updates_internally(self):
        self.risk.check_portfolio_drawdown(
            current_value=100.0,
            peak_value=100.0,
            max_drawdown_threshold=0.15,
        )
        self.risk.check_portfolio_drawdown(
            current_value=120.0,
            peak_value=120.0,
            max_drawdown_threshold=0.15,
        )

        approved, reason = self.risk.check_portfolio_drawdown(
            current_value=90.0,
            peak_value=100.0,
            max_drawdown_threshold=0.15,
        )
        self.assertFalse(approved)
        self.assertEqual(
            reason, "portfolio drawdown limit reached — manual reset required"
        )

    def test_bearish_signal_approved(self):
        approved, reason = self.risk.check(
            signal_value=-1,
            order_size_usd=5.0,
            current_exposure_usd=0.0,
            paper_mode=True,
        )
        self.assertTrue(approved)
        self.assertEqual(reason, "approved")


if __name__ == "__main__":
    unittest.main()
