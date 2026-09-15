"""Tests for equity/data/monitoring.py's semantic duplicate detection
(`_extract_key_signals`/`_semantically_similar`), the per-ticker daily cap
in `add_monitoring_items()`, and the plain-duplicate collapse in
`deduplicate_monitoring()`.
"""

import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from equity.data import monitoring
from equity.data.monitoring import (
    _extract_key_signals,
    _semantically_similar,
    add_monitoring_items,
    deduplicate_monitoring,
    load_monitoring,
)


class TestExtractKeySignals(unittest.TestCase):
    def test_extracts_dollar_level(self):
        self.assertIn("level_81.0", _extract_key_signals("TLT breaks below $81.00"))

    def test_extracts_bare_number_level(self):
        self.assertIn("level_29200.0", _extract_key_signals("NQ1 reclaims 29200"))

    def test_extracts_direction_word(self):
        signals = _extract_key_signals("Closes below $81.00")
        self.assertIn("dir_below", signals)
        self.assertIn("dir_close", signals)

    def test_extracts_time_condition(self):
        self.assertIn("time_consecutive", _extract_key_signals("three consecutive closes below $81"))


class TestSemanticallySimilar(unittest.TestCase):
    def test_paraphrase_with_same_level_and_direction_is_similar(self):
        a = "Three consecutive session closes below $81.00 — thesis-breaker trigger"
        b = "Count Tuesday session close — if below $81.00 this is consecutive close three"
        self.assertTrue(_semantically_similar(a, b))

    def test_same_direction_different_level_is_not_similar(self):
        # Low word overlap and different price levels — must not collide
        # just because both happen to mention "below".
        a = "TLT thesis-breaker if it settles below $81.00 for three sessions"
        b = "Watch for a Fed-driven selloff pushing yields below $75.00 support"
        self.assertFalse(_semantically_similar(a, b))

    def test_unrelated_items_are_not_similar(self):
        a = "TLT breaks below $81.00 for 3 sessions"
        b = "Watch Fed commentary on rate path"
        self.assertFalse(_semantically_similar(a, b))


class MonitoringStorageTestCase(unittest.TestCase):
    """Points MONITORING_PATH at a throwaway file for the duration of the test."""

    def setUp(self):
        self._tmpdir = TemporaryDirectory()
        tmp_path = Path(self._tmpdir.name) / "monitoring.json"
        self._patcher = patch.object(monitoring, "MONITORING_PATH", tmp_path)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        self._tmpdir.cleanup()


class TestAddMonitoringItemsDailyCap(MonitoringStorageTestCase):
    def test_caps_at_max_per_ticker_per_day(self):
        add_monitoring_items([
            {"ticker": "TLT", "item": "Closes below $81.00", "priority": "high"},
            {"ticker": "TLT", "item": "Yield spike above 5.5% on 10Y", "priority": "high"},
            {"ticker": "TLT", "item": "Fed minutes hawkish surprise", "priority": "medium"},
        ])
        items = [i for i in load_monitoring() if i["ticker"] == "TLT"]
        self.assertEqual(len(items), monitoring.MAX_PER_TICKER_PER_DAY)

    def test_different_tickers_each_get_their_own_cap(self):
        add_monitoring_items([
            {"ticker": "TLT", "item": "Closes below $81.00", "priority": "high"},
            {"ticker": "URA", "item": "Closes below $42.00", "priority": "high"},
        ])
        self.assertEqual(len({i["ticker"] for i in load_monitoring()}), 2)

    def test_semantic_duplicate_is_skipped_not_counted_against_cap(self):
        add_monitoring_items([{"ticker": "TLT", "item": "Closes below $81.00", "priority": "high"}])
        add_monitoring_items([{"ticker": "TLT", "item": "Close below $81.00 again", "priority": "high"}])
        items = [i for i in load_monitoring() if i["ticker"] == "TLT"]
        self.assertEqual(len(items), 1)


class TestDeduplicateMonitoringCollapsesParaphrases(MonitoringStorageTestCase):
    def test_collapses_plain_semantic_duplicates_with_no_supersession_signal(self):
        today_str = str(date.today())
        monitoring._save_all({
            "items": [
                {
                    "id": "tlt_1", "ticker": "TLT",
                    "item": "Three consecutive session closes below $81.00 — thesis-breaker trigger",
                    "priority": "high", "added_date": today_str, "added_from": "test",
                    "status": "active", "last_checked": today_str, "notes": [],
                },
                {
                    "id": "tlt_2", "ticker": "TLT",
                    "item": "Count Tuesday session close — if below $81.00 this is consecutive close three",
                    "priority": "high", "added_date": today_str, "added_from": "test",
                    "status": "active", "last_checked": today_str, "notes": [],
                },
            ]
        })
        resolved = deduplicate_monitoring()
        self.assertEqual(resolved, 1)
        self.assertEqual(len(load_monitoring()), 1)


if __name__ == "__main__":
    unittest.main()
