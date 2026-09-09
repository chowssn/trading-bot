"""Tests for equity/brief/brief_synthesizer.py's _parse_and_persist_monitoring()
— pure text-parsing logic, with equity.data.monitoring.add_monitoring_items()
mocked out so these never touch the real monitoring.json.
"""

import unittest
from unittest.mock import patch

from types import SimpleNamespace

from equity.brief import brief_synthesizer
from equity.brief.brief_synthesizer import (
    MONITORING_BLOCK_END,
    MONITORING_BLOCK_START,
    _emitted_empty_monitoring_block,
    _extract_text,
    _parse_and_persist_monitoring,
)


class TestParseAndPersistMonitoring(unittest.TestCase):
    def test_empty_text_persists_nothing(self):
        with patch("equity.data.monitoring.add_monitoring_items") as mock_add:
            _parse_and_persist_monitoring("")
        mock_add.assert_not_called()

    def test_no_header_persists_nothing(self):
        with patch("equity.data.monitoring.add_monitoring_items") as mock_add:
            _parse_and_persist_monitoring("SUMMARY\nJust a normal day, nothing notable.")
        mock_add.assert_not_called()

    def test_none_sentinel_persists_nothing(self):
        text = "SUMMARY\nQuiet day.\n\nNEW MONITORING ITEMS:\n(none)"
        with patch("equity.data.monitoring.add_monitoring_items") as mock_add:
            _parse_and_persist_monitoring(text)
        mock_add.assert_not_called()

    def test_standard_header_parses_pipe_delimited_items(self):
        text = (
            "SUMMARY\nText.\n\n"
            "NEW MONITORING ITEMS:\n"
            "TSLA | Cybercab NHTSA probe | high\n"
            "PLTR | RSI oversold watch below 35 | medium"
        )
        with patch("equity.data.monitoring.add_monitoring_items") as mock_add:
            _parse_and_persist_monitoring(text, source="test_source")
        mock_add.assert_called_once()
        items = mock_add.call_args[0][0]
        self.assertEqual([i["ticker"] for i in items], ["TSLA", "PLTR"])
        self.assertTrue(all(i["source"] == "test_source" for i in items))
        self.assertEqual(items[0]["priority"], "high")

    def test_bold_markdown_header_variant_parses(self):
        text = "**NEW MONITORING ITEMS**:\nUSDJPY | BOJ intervention risk near 160 | medium"
        with patch("equity.data.monitoring.add_monitoring_items") as mock_add:
            _parse_and_persist_monitoring(text)
        mock_add.assert_called_once()
        self.assertEqual(mock_add.call_args[0][0][0]["ticker"], "USDJPY")

    def test_new_items_to_monitor_header_variant_parses(self):
        text = "NEW ITEMS TO MONITOR:\nGOOGL | 200D MA test | high"
        with patch("equity.data.monitoring.add_monitoring_items") as mock_add:
            _parse_and_persist_monitoring(text)
        mock_add.assert_called_once()
        self.assertEqual(mock_add.call_args[0][0][0]["ticker"], "GOOGL")

    def test_invalid_priority_defaults_to_medium(self):
        text = "NEW MONITORING ITEMS:\nAAPL | some condition | urgent"
        with patch("equity.data.monitoring.add_monitoring_items") as mock_add:
            _parse_and_persist_monitoring(text)
        self.assertEqual(mock_add.call_args[0][0][0]["priority"], "medium")

    def test_non_pipe_line_in_block_is_skipped_not_misparsed(self):
        # A prose line inside the block (e.g. the model drifting back into
        # commentary) must not be misread as "TICKER: description" — see
        # _parse_and_persist_monitoring()'s docstring on why the old
        # colon/dash fallback heuristic was dropped.
        text = (
            "NEW MONITORING ITEMS:\n"
            "No items warrant tracking today given the quiet session.\n"
            "TSLA | Cybercab NHTSA probe | high"
        )
        with patch("equity.data.monitoring.add_monitoring_items") as mock_add:
            _parse_and_persist_monitoring(text)
        items = mock_add.call_args[0][0]
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["ticker"], "TSLA")

    def test_add_monitoring_items_failure_does_not_raise(self):
        text = "NEW MONITORING ITEMS:\nTSLA | condition | high"
        with patch("equity.data.monitoring.add_monitoring_items", side_effect=RuntimeError("boom")):
            _parse_and_persist_monitoring(text)  # must not raise


def _parsed_tickers(text: str) -> list[str]:
    with patch("equity.data.monitoring.add_monitoring_items") as mock_add:
        _parse_and_persist_monitoring(text, source="test")
    if not mock_add.called:
        return []
    return [i["ticker"] for i in mock_add.call_args[0][0]]


class TestDelimitedMonitoringBlock(unittest.TestCase):
    """The `<<<NEW_MONITORING_ITEMS>>>` sentinel the synthesis prompts now ask for."""

    def test_delimited_block_parses(self):
        text = (
            "SUMMARY: Test.\n\n"
            f"{MONITORING_BLOCK_START}\n"
            "TSLA | Q3 revenue vs 16M threshold | high\n"
            "COPPER | copper/gold ratio break | medium\n"
            f"{MONITORING_BLOCK_END}\n"
        )
        self.assertEqual(_parsed_tickers(text), ["TSLA", "COPPER"])

    def test_delimited_block_parses_when_end_marker_is_missing(self):
        # Response cut off before the closing marker — still recover the items.
        text = f"SUMMARY.\n\n{MONITORING_BLOCK_START}\nTSLA | Q3 revenue | high\n"
        self.assertEqual(_parsed_tickers(text), ["TSLA"])

    def test_none_inside_delimited_block_persists_nothing(self):
        text = f"SUMMARY.\n\n{MONITORING_BLOCK_START}\nNONE\n{MONITORING_BLOCK_END}\n"
        self.assertEqual(_parsed_tickers(text), [])

    def test_delimited_block_wins_over_a_stray_legacy_header(self):
        text = (
            "I will close with NEW MONITORING ITEMS.\n\n"
            f"{MONITORING_BLOCK_START}\n"
            "TSLA | real item | high\n"
            f"{MONITORING_BLOCK_END}\n"
        )
        self.assertEqual(_parsed_tickers(text), ["TSLA"])


class TestMonitoringBlockRecovery(unittest.TestCase):
    """Formatting variants that the old first-match-wins parser dropped."""

    def test_blank_line_between_items_does_not_truncate_the_block(self):
        # Regression: the old `(?:\n\n|\Z)` terminator ended the block at the
        # first blank line, so only the first item survived.
        text = "S.\n\nNEW MONITORING ITEMS:\nTSLA | a | high\n\nCOPPER | b | low\n"
        self.assertEqual(_parsed_tickers(text), ["TSLA", "COPPER"])

    def test_header_named_in_a_preamble_does_not_shadow_the_real_block(self):
        # Regression: the first match was an empty capture from the preamble
        # mention, and first-match-wins meant the real items below were never
        # looked for.
        text = (
            "I will end with NEW MONITORING ITEMS and SUGGESTIONS.\n\n"
            "NEW MONITORING ITEMS:\nTSLA | a | high\nCOPPER | b | low\n"
        )
        self.assertEqual(_parsed_tickers(text), ["TSLA", "COPPER"])

    def test_bold_header_parses(self):
        text = "S.\n\n**NEW MONITORING ITEMS**\nTSLA | a | high\nCOPPER | b | low\n"
        self.assertEqual(_parsed_tickers(text), ["TSLA", "COPPER"])

    def test_bulleted_and_bolded_items_parse(self):
        text = "S.\n\n**NEW MONITORING ITEMS**\n- **TSLA** | a | high\n- **COPPER** | b | low\n"
        self.assertEqual(_parsed_tickers(text), ["TSLA", "COPPER"])

    def test_markdown_table_parses_without_header_or_separator_rows(self):
        # The model sometimes renders items as a table. Its `|---|---|`
        # separator must not terminate the block, and its header row must not
        # become a monitoring item for a ticker named "TICKER".
        text = (
            "S.\n\nNEW MONITORING ITEMS:\n"
            "| Ticker | Condition | Priority |\n"
            "|---|---|---|\n"
            "| TSLA | Q3 revenue | high |\n"
            "| COPPER | ratio break | medium |\n"
        )
        self.assertEqual(_parsed_tickers(text), ["TSLA", "COPPER"])

    def test_horizontal_rule_still_ends_the_block(self):
        text = (
            "S.\n\nNEW MONITORING ITEMS:\n"
            "TSLA | a | high\n"
            "---\n"
            "SUGGESTIONS: not an item | at all | here\n"
        )
        self.assertEqual(_parsed_tickers(text), ["TSLA"])

    def test_truncated_response_with_no_block_persists_nothing(self):
        # A response cut off at max_tokens before reaching the block.
        text = "SUMMARY.\n\n**MONITORING UPDATES**\n\n*(No active items carried into today - see new"
        self.assertEqual(_parsed_tickers(text), [])


class TestEmptyBlockDetection(unittest.TestCase):
    def test_none_block_is_recognized_as_deliberate(self):
        text = f"{MONITORING_BLOCK_START}\nNONE\n{MONITORING_BLOCK_END}"
        self.assertTrue(_emitted_empty_monitoring_block(text))

    def test_populated_block_is_not_empty(self):
        text = f"{MONITORING_BLOCK_START}\nTSLA | a | high\n{MONITORING_BLOCK_END}"
        self.assertFalse(_emitted_empty_monitoring_block(text))

    def test_absent_block_is_not_a_deliberate_empty(self):
        # No block at all is a failure, not a "nothing to track" answer.
        self.assertFalse(_emitted_empty_monitoring_block("SUMMARY: quiet day."))


class TestTruncationDetection(unittest.TestCase):
    def test_max_tokens_stop_is_flagged(self):
        response = SimpleNamespace(
            stop_reason="max_tokens",
            content=[SimpleNamespace(text="truncated body")],
        )
        with self.assertLogs("equity.brief.brief_synthesizer", level="WARNING") as logs:
            text = _extract_text(response, "Test synthesis")
        self.assertEqual(text, "truncated body")
        self.assertTrue(any("max_tokens" in line for line in logs.output))

    def test_normal_stop_is_not_flagged(self):
        response = SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(text="complete body")],
        )
        with patch.object(brief_synthesizer.logger, "warning") as mock_warn:
            text = _extract_text(response, "Test synthesis")
        self.assertEqual(text, "complete body")
        mock_warn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
