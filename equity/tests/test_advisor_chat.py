"""Tests for equity/telegram/advisor.py's Advisor.chat() auto-continuation —
when a response is cut off by Claude's max_tokens limit, chat() should
transparently request a continuation and append it, rather than returning
(or persisting to thread history) a truncated answer. The Claude client and
ThreadManager are mocked throughout.
"""

import unittest
from unittest.mock import MagicMock, patch

from equity.telegram.advisor import Advisor


def _make_advisor() -> Advisor:
    thread_manager = MagicMock()
    # chat() does `messages + [...]` when building the continuation request,
    # so this needs to be a real list, not the MagicMock default.
    thread_manager.get_messages_for_api.return_value = [
        {"role": "user", "content": "Long question"}
    ]
    return Advisor(api_key="test-key", thread_manager=thread_manager)


def _text_block(text: str):
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _response(stop_reason: str, text: str, input_tokens: int, output_tokens: int):
    response = MagicMock()
    response.stop_reason = stop_reason
    response.content = [_text_block(text)]
    response.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
    return response


class TestChatContinuation(unittest.TestCase):
    def test_chat_continues_on_max_tokens(self):
        """When chat() gets stop_reason=max_tokens, it should automatically
        send a continuation request and append the result to the original."""
        adv = _make_advisor()
        first_response = _response("max_tokens", "Partial response here", 1000, 2000)
        cont_response = _response("end_turn", "...continuation of the response.", 500, 300)

        with patch.object(
            adv.client.messages, "create", side_effect=[first_response, cont_response]
        ) as mock_create:
            result = adv.chat("test_thread", "Long question", "system")

        self.assertIn("Partial response here", result)
        self.assertIn("continuation of the response", result)
        self.assertEqual(mock_create.call_count, 2)

    def test_chat_returns_partial_if_continuation_fails(self):
        """If the continuation call fails, the original truncated response
        should still be returned rather than raising."""
        adv = _make_advisor()
        first_response = _response("max_tokens", "Partial response", 1000, 2000)

        with patch.object(
            adv.client.messages,
            "create",
            side_effect=[first_response, Exception("API error on continuation")],
        ) as mock_create:
            result = adv.chat("test_thread", "Long question", "system")

        self.assertEqual(result, "Partial response")
        self.assertEqual(mock_create.call_count, 2)

    def test_chat_does_not_continue_when_not_truncated(self):
        """A normal (non-truncated) response should only cost one API call."""
        adv = _make_advisor()
        response = _response("end_turn", "Complete response", 1000, 200)

        with patch.object(adv.client.messages, "create", return_value=response) as mock_create:
            result = adv.chat("test_thread", "Short question", "system")

        self.assertEqual(result, "Complete response")
        mock_create.assert_called_once()


if __name__ == "__main__":
    unittest.main()
