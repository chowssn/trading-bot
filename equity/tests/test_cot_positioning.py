"""Tests for market_snapshot.py's CFTC COT parsing and formatting.

The sample rows below are trimmed from the real 2026-10-06 FinFutWk.txt:
column positions and spacing are preserved exactly, since the file has no
header row and is read positionally.
"""

import unittest

from equity.brief import market_snapshot
from equity.config.market_config import COT_PCT_OI_EXTREME


def _row(market: str, oi: int, d_l: int, d_s: int, am_l: int, am_s: int, lev_l: int, lev_s: int) -> str:
    """One FinFutWk.txt line: market, dates, codes, then the position columns."""
    cols = [
        f'"{market}"', "261006", "2026-10-06", "043602", "CBT ", "00", "020 ",
        str(oi), str(d_l), str(d_s), "0", str(am_l), str(am_s), "0",
        str(lev_l), str(lev_s), "0",
    ]
    return ",".join(cols)


# Real figures from the 2026-10-06 report.
_SAMPLE = "\n".join([
    _row("UST 10Y NOTE - CHICAGO BOARD OF TRADE", 5742990, 98637, 740670, 3602149, 774010, 510611, 2650396),
    _row("UST BOND - CHICAGO BOARD OF TRADE", 1898067, 44875, 237552, 1096523, 690895, 172246, 361924),
    _row("ULTRA UST BOND - CHICAGO BOARD OF TRADE", 2404377, 21456, 110211, 1507861, 752102, 58345, 784602),
    _row("UST 2Y NOTE - CHICAGO BOARD OF TRADE", 4485811, 98563, 607014, 2471312, 792225, 693301, 1950785),
    _row("UST 5Y NOTE - CHICAGO BOARD OF TRADE", 6758624, 83428, 919320, 4218531, 1304245, 541071, 2668638),
    _row("CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE", 371048, 189180, 69223, 97612, 133418, 31682, 111307),
])


class TestParseCotFile(unittest.TestCase):
    def test_finds_all_four_benchmark_tenors(self):
        result = market_snapshot._parse_cot_file(_SAMPLE)
        self.assertEqual(set(result["positions"]), {"2Y", "5Y", "10Y", "30Y"})

    def test_report_date(self):
        self.assertEqual(market_snapshot._parse_cot_file(_SAMPLE)["report_date"], "2026-10-06")

    def test_net_positions(self):
        pos = market_snapshot._parse_cot_file(_SAMPLE)["positions"]["10Y"]
        self.assertEqual(pos["lev_net"], 510611 - 2650396)
        self.assertEqual(pos["am_net"], 3602149 - 774010)
        self.assertEqual(pos["dealer_net"], 98637 - 740670)
        self.assertEqual(pos["open_interest"], 5742990)

    def test_ultra_bond_does_not_overwrite_30y(self):
        """'UST BOND - ...' is a substring of 'ULTRA UST BOND - ...' — exact
        matching must keep ZB's numbers out of UB's row and vice versa."""
        pos = market_snapshot._parse_cot_file(_SAMPLE)["positions"]["30Y"]
        self.assertEqual(pos["open_interest"], 1898067)       # ZB, not UB's 2404377
        self.assertEqual(pos["lev_net"], 172246 - 361924)

    def test_non_treasury_rows_ignored(self):
        result = market_snapshot._parse_cot_file(_SAMPLE)
        self.assertNotIn("CANADIAN", str(result["positions"].keys()))
        self.assertEqual(len(result["positions"]), 4)

    def test_pct_of_open_interest(self):
        pos = market_snapshot._parse_cot_file(_SAMPLE)["positions"]["10Y"]
        self.assertAlmostEqual(pos["lev_pct_oi"], (510611 - 2650396) / 5742990, places=6)

    def test_no_header_row_assumption(self):
        """A header row, if CFTC ever adds one, must be skipped, not parsed."""
        header = "Market_and_Exchange_Names,As_of_Date,Report_Date,a,b,c,d,e,f,g,h,i,j,k,l,m,n"
        result = market_snapshot._parse_cot_file(header + "\n" + _SAMPLE)
        self.assertEqual(len(result["positions"]), 4)

    def test_empty_input(self):
        self.assertEqual(market_snapshot._parse_cot_file(""), {"positions": {}, "report_date": None})

    def test_short_rows_skipped(self):
        self.assertEqual(market_snapshot._parse_cot_file("a,b,c\n")["positions"], {})

    def test_withheld_values_skip_row_rather_than_count_as_zero(self):
        withheld = _row("UST 10Y NOTE - CHICAGO BOARD OF TRADE", 100, 1, 1, 1, 1, 1, 1).replace(
            ",1,1,0,1,1,0,1,1,0", ",1,1,0,.,.,0,1,1,0"
        )
        self.assertEqual(market_snapshot._parse_cot_file(withheld)["positions"], {})


class TestCotIntHelper(unittest.TestCase):
    def test_blank_and_withheld_are_none(self):
        self.assertIsNone(market_snapshot._cot_int(["", "."], 0))
        self.assertIsNone(market_snapshot._cot_int(["", "."], 1))

    def test_out_of_range_is_none(self):
        self.assertIsNone(market_snapshot._cot_int(["1"], 5))

    def test_strips_padding_and_commas(self):
        self.assertEqual(market_snapshot._cot_int(["  1,234  "], 0), 1234)

    def test_zero_is_zero_not_none(self):
        self.assertEqual(market_snapshot._cot_int(["0"], 0), 0)

    def test_negative(self):
        self.assertEqual(market_snapshot._cot_int(["-500"], 0), -500)


class TestFormatCotPositioning(unittest.TestCase):
    def setUp(self):
        self.data = market_snapshot._parse_cot_file(_SAMPLE)

    def test_empty_data_returns_empty_string(self):
        self.assertEqual(market_snapshot.format_cot_positioning({}), "")
        self.assertEqual(market_snapshot.format_cot_positioning(None), "")
        self.assertEqual(market_snapshot.format_cot_positioning({"positions": {}}), "")

    def test_includes_all_tenors_in_curve_order(self):
        text = market_snapshot.format_cot_positioning(self.data)
        order = [text.index(f"*{t} Treasury*") for t in ("2Y", "5Y", "10Y", "30Y")]
        self.assertEqual(order, sorted(order))

    def test_report_date_in_header(self):
        self.assertIn("2026-10-06", market_snapshot.format_cot_positioning(self.data))

    def test_crowded_flag_on_10y_only(self):
        """10Y lev net is -37% of OI; 30Y is -10%, below both thresholds."""
        text = market_snapshot.format_cot_positioning(self.data)
        tenor_blocks = dict(
            zip(("2Y", "5Y", "10Y", "30Y"), [
                text.split(f"*{t} Treasury*")[1].split("\n")[0]
                for t in ("2Y", "5Y", "10Y", "30Y")
            ])
        )
        self.assertIn("CROWDED", tenor_blocks["10Y"])
        self.assertNotIn("CROWDED", tenor_blocks["30Y"])

    def test_basis_trade_caveat_present_when_interpretation_fires(self):
        self.assertIn("basis trade", market_snapshot.format_cot_positioning(self.data))

    def test_divergence_line_when_real_money_long_and_spec_short(self):
        self.assertIn("asset managers are buying", market_snapshot.format_cot_positioning(self.data))


class TestCotInterpretation(unittest.TestCase):
    def test_no_interpretation_without_10y_or_30y(self):
        self.assertEqual(market_snapshot._format_cot_interpretation({}), [])

    def test_short_base_ratio_uses_magnitudes_not_signed_comparison(self):
        """Both legs net LONG must not produce a short-covering read."""
        positions = {
            "10Y": {"lev_net": 2_000_000, "am_net": 100, "lev_pct_oi": 0.4},
            "30Y": {"lev_net": 100_000, "am_net": 100, "lev_pct_oi": 0.1},
        }
        lines = market_snapshot._format_cot_interpretation(positions)
        self.assertFalse(any("covering" in line for line in lines))

    def test_short_base_ratio_fires_when_both_short_and_30y_much_smaller(self):
        positions = {
            "10Y": {"lev_net": -2_000_000, "am_net": 100, "lev_pct_oi": -0.4},
            "30Y": {"lev_net": -100_000, "am_net": 100, "lev_pct_oi": -0.1},
        }
        lines = market_snapshot._format_cot_interpretation(positions)
        self.assertTrue(any("covering" in line for line in lines))

    def test_no_crowding_line_below_threshold(self):
        positions = {"10Y": {"lev_net": -100, "am_net": -100, "lev_pct_oi": -0.01}}
        self.assertEqual(market_snapshot._format_cot_interpretation(positions), [])

    def test_crowding_line_at_extreme_threshold(self):
        positions = {"10Y": {"lev_net": -100, "am_net": -100, "lev_pct_oi": -COT_PCT_OI_EXTREME}}
        lines = market_snapshot._format_cot_interpretation(positions)
        self.assertTrue(any("short base" in line for line in lines))

    def test_missing_pct_oi_does_not_crash(self):
        positions = {"10Y": {"lev_net": -100, "am_net": 100}}
        self.assertIsInstance(market_snapshot._format_cot_interpretation(positions), list)


if __name__ == "__main__":
    unittest.main()
