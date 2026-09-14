"""Tests for equity/telegram/advisor.py's Advisor.summarize_messages() —
the nightly per-thread summarization call. Covers the per-message (300
char) and total (SUMMARIZE_MAX_CHARS) truncation applied before the
transcript is sent to Claude; the Claude client is mocked throughout.
"""

import unittest
from unittest.mock import MagicMock, patch

from equity.telegram.advisor import SUMMARIZE_MAX_CHARS, Advisor


def _make_advisor() -> Advisor:
    return Advisor(api_key="test-key", thread_manager=MagicMock())


def _text_block(text: str):
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


class TestSummarizeMessages(unittest.TestCase):
    def test_empty_messages_returns_empty_string_without_calling_claude(self):
        adv = _make_advisor()
        with patch.object(adv.client.messages, "create") as mock_create:
            result = adv.summarize_messages([])
        self.assertEqual(result, "")
        mock_create.assert_not_called()

    def test_per_message_truncated_to_300_chars(self):
        adv = _make_advisor()
        long_message = "x" * 1000
        response = MagicMock(content=[_text_block("summary")])
        with patch.object(adv.client.messages, "create", return_value=response) as mock_create:
            adv.summarize_messages([{"role": "user", "content": long_message}])
        sent_text = mock_create.call_args.kwargs["messages"][0]["content"]
        # 300 chars of "x" survive; the rest of the 1000-char message doesn't.
        self.assertIn("x" * 300, sent_text)
        self.assertNotIn("x" * 301, sent_text)

    def test_total_transcript_capped_at_summarize_max_chars(self):
        adv = _make_advisor()
        # Many short messages whose 300-char-capped total still exceeds
        # SUMMARIZE_MAX_CHARS — the safety net, not the per-message cap,
        # is what should kick in here.
        messages = [{"role": "user", "content": "y" * 300} for _ in range(50)]
        response = MagicMock(content=[_text_block("summary")])
        with patch.object(adv.client.messages, "create", return_value=response) as mock_create:
            adv.summarize_messages(messages)
        sent_text = mock_create.call_args.kwargs["messages"][0]["content"]
        self.assertIn("[truncated for summarization]", sent_text)
        # Allow for the "Discussion:\n" prefix and the truncation marker.
        self.assertLess(len(sent_text), SUMMARIZE_MAX_CHARS + 200)

    def test_short_transcript_is_not_marked_truncated(self):
        adv = _make_advisor()
        response = MagicMock(content=[_text_block("summary")])
        with patch.object(adv.client.messages, "create", return_value=response) as mock_create:
            adv.summarize_messages([{"role": "user", "content": "short message"}])
        sent_text = mock_create.call_args.kwargs["messages"][0]["content"]
        self.assertNotIn("[truncated for summarization]", sent_text)

    def test_api_exception_returns_placeholder_not_raise(self):
        adv = _make_advisor()
        with patch.object(adv.client.messages, "create", side_effect=RuntimeError("boom")):
            result = adv.summarize_messages([{"role": "user", "content": "hi"}])
        self.assertEqual(result, "[Summary unavailable — Claude API error]")


if __name__ == "__main__":
    unittest.main()
