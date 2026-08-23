"""Validates `roic_calculator.calculate_roic()` against hardcoded Bloomberg values.

Runs the ROIC calculation for MSFT and ADBE, prints a detailed component
breakdown for each, and flags the deviation from Bloomberg Terminal's
RETURN_ON_INV_CAPITAL (current and 5Y average) as GREEN/YELLOW/RED.

Bloomberg reference values are hardcoded below — this is a point-in-time
validation snapshot, not a live lookup. Re-pull from the Terminal and update
`BLOOMBERG_VALUES` if you re-run this after a new fiscal period closes.

Usage: python -m equity.tools.bloomberg_validation_v2
"""

import logging
import os

from dotenv import load_dotenv

from equity.tools.roic_calculator import calculate_roic

logger = logging.getLogger(__name__)

TICKERS = ["MSFT", "ADBE"]

# Bloomberg Terminal RETURN_ON_INV_CAPITAL, hardcoded for validation.
BLOOMBERG_VALUES = {
    "MSFT": {"roic_bbg": 25.02, "roic_5y_avg_bbg": 28.11},
    "ADBE": {"roic_bbg": 38.92, "roic_5y_avg_bbg": 27.25},
}

# Deviation thresholds, in absolute percentage points of ROIC.
GREEN_THRESHOLD_PP = 3.0
YELLOW_THRESHOLD_PP = 8.0

_ANSI_GREEN = "\033[92m"
_ANSI_YELLOW = "\033[93m"
_ANSI_RED = "\033[91m"
_ANSI_RESET = "\033[0m"


def _flag(deviation_pp: float) -> str:
    """GREEN/YELLOW/RED label (with ANSI color) for an absolute pp deviation."""
    if deviation_pp <= GREEN_THRESHOLD_PP:
        return f"{_ANSI_GREEN}GREEN{_ANSI_RESET}"
    if deviation_pp <= YELLOW_THRESHOLD_PP:
        return f"{_ANSI_YELLOW}YELLOW{_ANSI_RESET}"
    return f"{_ANSI_RED}RED{_ANSI_RESET}"


def _print_component_breakdown(result: dict) -> None:
    current = result["components"]["current_period"]
    print(f"\n--- Component breakdown: {result['ticker']} ---")
    print(f"  Current period (T12M source: {current['t12m_source']}, quarters: {current['quarters_used']})")
    print(f"    Operating income (T12M):     ${current['operating_income_t12m_musd']:,.1f}M")
    print(f"    Income tax expense (T12M):   ${current['income_tax_expense_t12m_musd']:,.1f}M")
    print(f"    Income before tax (T12M):    ${current['income_before_tax_t12m_musd']:,.1f}M")
    print(f"    Effective tax rate:          {current['effective_tax_rate_pct']:.2f}%")
    print(f"    -> NOPAT:                    ${current['nopat_musd']:,.1f}M")
    print(f"    Invested capital (current):  ${current['invested_capital_musd']:,.1f}M")
    print(f"    Avg invested capital:        ${current['avg_invested_capital_musd']:,.1f}M")
    print(f"    -> ROIC (current):           {result['roic_current']:.2f}%")

    print("  Annual history used for roic_5y_avg:")
    for r in result["components"]["annual"]:
        print(
            f"    {r['date']}: NOPAT=${r['nopat_musd']:,.1f}M / "
            f"avg IC=${r['avg_invested_capital_musd']:,.1f}M "
            f"(tax rate {r['effective_tax_rate_pct']:.2f}%) -> ROIC={r['roic_pct']:.2f}%"
        )
    print(f"    -> 5Y average ROIC: {result['roic_5y_avg']:.2f}%")


def _print_deviation(ticker: str, result: dict) -> None:
    bbg = BLOOMBERG_VALUES[ticker]
    dev_current = abs(result["roic_current"] - bbg["roic_bbg"])
    dev_5y = abs(result["roic_5y_avg"] - bbg["roic_5y_avg_bbg"])

    print(f"\n  Bloomberg comparison: {ticker}")
    print(
        f"    ROIC current:  ours={result['roic_current']:.2f}%  bbg={bbg['roic_bbg']:.2f}%  "
        f"deviation={dev_current:.2f}pp  [{_flag(dev_current)}]"
    )
    print(
        f"    ROIC 5Y avg:   ours={result['roic_5y_avg']:.2f}%  bbg={bbg['roic_5y_avg_bbg']:.2f}%  "
        f"deviation={dev_5y:.2f}pp  [{_flag(dev_5y)}]"
    )


def _print_methodology_notes(result: dict) -> None:
    print(f"\n  Methodology notes ({result['ticker']}):")
    for note in result["methodology_notes"]:
        print(f"    - {note}")


def run_validation() -> dict:
    load_dotenv()
    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        raise RuntimeError("FMP_API_KEY not set — add it to .env")

    results = {}
    for ticker in TICKERS:
        logger.info("Calculating ROIC for %s...", ticker)
        results[ticker] = calculate_roic(ticker, api_key, periods=5)
    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=== ROIC Calculator vs. Bloomberg Terminal Validation ===")
    results = run_validation()

    for ticker in TICKERS:
        result = results[ticker]
        _print_component_breakdown(result)
        _print_deviation(ticker, result)
        _print_methodology_notes(result)

    print(f"\nThresholds: GREEN <= {GREEN_THRESHOLD_PP:.1f}pp, YELLOW <= {YELLOW_THRESHOLD_PP:.1f}pp, RED > {YELLOW_THRESHOLD_PP:.1f}pp")
    print("Validation complete.")


if __name__ == "__main__":
    main()
