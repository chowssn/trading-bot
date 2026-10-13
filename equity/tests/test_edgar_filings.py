"""Tests for news_triage.py's SEC EDGAR filing helpers.

Network calls are stubbed — these cover the parsing, URL construction and
filtering, which is where the bugs live. The submissions-API payload shape
below matches a real `data.sec.gov/submissions/CIK*.json` response.
"""

import unittest
from unittest.mock import patch

from equity.portfolio import news_triage


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


_SUBMISSIONS = {
    "cik": "1318605",
    "filings": {
        "recent": {
            "form": ["8-K", "4", "10-Q", "424B5", "8-K"],
            "filingDate": ["2026-10-10", "2026-10-09", "2026-10-08", "2026-10-07", "2026-01-02"],
            "reportDate": ["2026-10-10", "2026-10-07", "2026-09-30", "", "2026-01-02"],
            "accessionNumber": [
                "0001628280-26-064366", "0001628280-26-064000",
                "0001628280-26-063000", "0001628280-26-062000",
                "0001628280-26-000111",
            ],
            "primaryDocument": [
                "tsla-20261002.htm", "xslF345X06/form4.xml",
                "tsla-10q.htm", "prospectus.htm", "old-8k.htm",
            ],
            "items": ["2.02,9.01", "", "", "", "8.01"],
        }
    },
}


class TestDescribe8kItems(unittest.TestCase):
    def test_known_code_described(self):
        self.assertEqual(news_triage._describe_8k_items("2.02"), "results of operations")

    def test_unknown_codes_dropped(self):
        """9.01 (exhibits) rides along on most 8-Ks and carries no signal."""
        self.assertEqual(news_triage._describe_8k_items("2.02,9.01"), "results of operations")

    def test_all_unknown_is_empty(self):
        self.assertEqual(news_triage._describe_8k_items("9.01"), "")

    def test_multiple_known_codes_joined(self):
        out = news_triage._describe_8k_items("1.01,1.02")
        self.assertIn("material agreement entered", out)
        self.assertIn("material agreement terminated", out)

    def test_duplicates_collapsed(self):
        self.assertEqual(news_triage._describe_8k_items("2.02,2.02"), "results of operations")

    def test_whitespace_tolerated(self):
        self.assertEqual(news_triage._describe_8k_items(" 2.02 , 9.01 "), "results of operations")


class TestFetchTickerFilings(unittest.TestCase):
    def _fetch(self, cutoff="2026-10-01"):
        with patch.object(news_triage.requests, "get", return_value=_Resp(_SUBMISSIONS)):
            return news_triage._fetch_ticker_filings("TSLA", 1318605, cutoff)

    def test_only_forms_of_interest(self):
        """10-Q is not a supply/demand event and must be filtered out."""
        self.assertNotIn("10-Q", [f["form"] for f in self._fetch()])

    def test_respects_cutoff(self):
        """`recent` is newest-first, so the January 8-K is past the cutoff."""
        self.assertNotIn("2026-01-02", [f["date"] for f in self._fetch()])

    def test_caps_per_ticker(self):
        self.assertLessEqual(len(self._fetch()), news_triage.EDGAR_MAX_FILINGS_PER_TICKER)

    def test_url_strips_accession_dashes_and_uses_unpadded_cik(self):
        url = self._fetch()[0]["url"]
        self.assertIn("/data/1318605/000162828026064366/", url)
        self.assertNotIn("0001628280-26-064366", url)

    def test_form_label_attached(self):
        self.assertEqual(self._fetch()[0]["form_label"], "material event")

    def test_items_described_for_8k(self):
        self.assertEqual(self._fetch()[0]["items"], "results of operations")

    def test_fetch_failure_returns_empty_list(self):
        with patch.object(news_triage.requests, "get", side_effect=RuntimeError("boom")):
            self.assertEqual(news_triage._fetch_ticker_filings("TSLA", 1318605, "2026-10-01"), [])

    def test_missing_primary_document_falls_back_to_browse_url(self):
        payload = {"filings": {"recent": dict(_SUBMISSIONS["filings"]["recent"])}}
        payload["filings"]["recent"]["primaryDocument"] = ["", "", "", "", ""]
        with patch.object(news_triage.requests, "get", return_value=_Resp(payload)):
            url = news_triage._fetch_ticker_filings("TSLA", 1318605, "2026-10-01")[0]["url"]
        self.assertIn("browse-edgar", url)


class TestFetchEdgarFilings(unittest.TestCase):
    def test_empty_ticker_list(self):
        self.assertEqual(news_triage.fetch_edgar_filings([]), {})

    def test_no_ticker_map_returns_empty(self):
        with patch.object(news_triage, "_fetch_edgar_ticker_map", return_value={}), \
             patch.object(news_triage, "_cache_age_hours", return_value=None):
            self.assertEqual(news_triage.fetch_edgar_filings(["TSLA"]), {})

    def test_unmapped_ticker_skipped_not_crashed(self):
        """ETFs (QQQ, TLT) have no ticker-mapped CIK — a normal skip."""
        with patch.object(news_triage, "_fetch_edgar_ticker_map", return_value={"TSLA": 1318605}), \
             patch.object(news_triage, "_cache_age_hours", return_value=None), \
             patch.object(news_triage, "_fetch_ticker_filings", return_value=[]), \
             patch.object(news_triage.time, "sleep"):
            self.assertEqual(news_triage.fetch_edgar_filings(["QQQ", "TSLA"]), {})

    def test_caps_ticker_count(self):
        seen = []

        def _spy(ticker, cik, cutoff):
            seen.append(ticker)
            return []

        many = [f"T{i}" for i in range(30)]
        with patch.object(news_triage, "_fetch_edgar_ticker_map", return_value={t: 1 for t in many}), \
             patch.object(news_triage, "_cache_age_hours", return_value=None), \
             patch.object(news_triage, "_fetch_ticker_filings", side_effect=_spy), \
             patch.object(news_triage.time, "sleep"):
            news_triage.fetch_edgar_filings(many)
        self.assertEqual(len(seen), news_triage.EDGAR_MAX_TICKERS)

    def test_tickers_uppercased(self):
        with patch.object(news_triage, "_fetch_edgar_ticker_map", return_value={"TSLA": 1318605}), \
             patch.object(news_triage, "_cache_age_hours", return_value=None), \
             patch.object(news_triage, "_fetch_ticker_filings", return_value=[{"form": "8-K"}]), \
             patch.object(news_triage.time, "sleep"):
            self.assertIn("TSLA", news_triage.fetch_edgar_filings(["tsla"]))


class TestFormatEdgarFilings(unittest.TestCase):
    def test_empty_returns_empty_string(self):
        self.assertEqual(news_triage.format_edgar_filings({}), "")

    def test_renders_linked_filing(self):
        data = {"TSLA": [{
            "form": "8-K", "form_label": "material event", "date": "2026-10-10",
            "items": "results of operations", "url": "https://example.com/x.htm",
        }]}
        text = news_triage.format_edgar_filings(data)
        self.assertIn("TSLA", text)
        self.assertIn("[8-K (material event)](https://example.com/x.htm)", text)
        self.assertIn("results of operations", text)

    def test_header_reflects_requested_window(self):
        data = {"T": [{"form": "4", "form_label": "insider transaction",
                       "date": "2026-10-10", "items": "", "url": "u"}]}
        self.assertIn("last 14 days", news_triage.format_edgar_filings(data, days_back=14))

    def test_no_trailing_dash_when_no_items(self):
        data = {"T": [{"form": "4", "form_label": "insider transaction",
                       "date": "2026-10-10", "items": "", "url": "u"}]}
        self.assertNotIn("—", news_triage.format_edgar_filings(data))


if __name__ == "__main__":
    unittest.main()
