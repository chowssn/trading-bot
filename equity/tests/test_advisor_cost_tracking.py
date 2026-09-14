"""Tests for equity/telegram/advisor.py's approximate API cost tracking
(_track_api_cost, _session_api_calls) — surfaced in /status as a rough
session spend estimate. Sonnet 4.6 pricing only; see _track_api_cost's
docstring for what this counter does and doesn't cover.
"""

import unittest
from unittest.mock import MagicMock

from equity.telegram import advisor as advisor_module
from equity.telegram.advisor import _track_api_cost


def _fake_response(input_tokens: int, output_tokens: int) -> MagicMock:
    response = MagicMock()
    response.usage.input_tokens = input_tokens
    response.usage.output_tokens = output_tokens
    return response


class TestTrackApiCost(unittest.TestCase):
    def setUp(self):
        # Module-level counters are shared state — reset between tests.
        advisor_module._session_api_calls["count"] = 0
        advisor_module._session_api_calls["est_cost_usd"] = 0.0

    def test_increments_call_count(self):
        _track_api_cost(_fake_response(1000, 500))
        _track_api_cost(_fake_response(1000, 500))
        self.assertEqual(advisor_module._session_api_calls["count"], 2)

    def test_computes_cost_at_sonnet_4_6_rates(self):
        # 1M input tokens @ $3/MTok + 1M output tokens @ $15/MTok = $18.
        _track_api_cost(_fake_response(1_000_000, 1_000_000))
        self.assertAlmostEqual(advisor_module._session_api_calls["est_cost_usd"], 18.0)

    def test_accumulates_across_calls(self):
        _track_api_cost(_fake_response(1_000_000, 0))  # $3
        _track_api_cost(_fake_response(0, 1_000_000))  # $15
        self.assertAlmostEqual(advisor_module._session_api_calls["est_cost_usd"], 18.0)

    def test_malformed_response_is_ignored_not_raised(self):
        bad_response = object()  # no .usage at all
        _track_api_cost(bad_response)  # must not raise
        self.assertEqual(advisor_module._session_api_calls["count"], 0)


if __name__ == "__main__":
    unittest.main()
