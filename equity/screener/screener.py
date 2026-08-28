"""Main orchestrator for the equity screener funnel.

Ties the three funnel stages together:

    universe.get_universe_tickers()   Russell 1000, from the IWB holdings CSV
        -> price_filter.run_price_filter()   cheap yfinance price/RSI/volume
           dislocation + market-cap pre-filter, ~1000 tickers -> ~30-80
              -> quality_scorer.score_ticker()   FMP/yfinance fundamentals,
                 0-100 quality score, per surviving ticker
                 -> timing_signal.classify_timing()   RSI-based entry timing

`run_screener()` returns the top `settings.MAX_SCREENER_OUTPUT` names that
clear `settings.MIN_QUALITY_SCORE` and carry no red flags, sorted ENTRY
before WATCH before WAIT, quality_score descending within each group.
`format_screener_output()` renders that DataFrame as a Telegram-ready string.
"""

import logging
import sys
from datetime import date
from pprint import pprint

import pandas as pd

from equity.config import settings
from equity.screener import price_filter, quality_scorer, timing_signal, universe

logger = logging.getLogger(__name__)

_SIGNAL_ORDER = {"ENTRY": 0, "WATCH": 1, "WAIT": 2}
_SIGNAL_HEADERS = {
    "ENTRY": "🟢 ENTRY (RSI turning)",
    "WATCH": "👁 WATCH (Oversold, not yet turning)",
    "WAIT": "⏳ WAIT (Too early)",
}
_DIVIDER = "━━━━━━━━━━━━━━━━━━━━━━━━"
_RSI_ARROWS = {"rising": "↑", "falling": "↓", "neutral": "→"}

RESULT_COLUMNS = quality_scorer.RESULT_KEYS + [
    "timing_signal", "timing_note", "return_1y", "rsi_14d", "rsi_14d_direction", "market_cap_b",
]


def _clear_todays_cache() -> None:
    """Delete today's price-filter and quality-scorer cache files.

    Neither `price_filter.run_price_filter()` nor `quality_scorer.score_ticker()`
    takes a `force_refresh` argument — both cache purely on file mtime/TTL.
    To honor `run_screener(force_refresh=True)` without changing either
    module's API, we remove today's cache files up front so the normal
    cache-miss path in each module re-fetches.
    """
    today = date.today().isoformat()
    removed = 0
    for path in quality_scorer.CACHE_DIR.glob(f"*_{today}.*"):
        try:
            path.unlink()
            removed += 1
        except OSError as exc:
            logger.warning("Could not remove cache file %s: %s", path, exc)
    logger.info("force_refresh: cleared %d cache file(s) for %s", removed, today)


def run_screener(force_refresh: bool = False) -> pd.DataFrame:
    """Run the full screener funnel and return the ranked results DataFrame.

    If `force_refresh` is True, today's price-filter/quality-scorer cache
    files are cleared and the Russell 1000 universe is force-refreshed
    before the funnel runs — otherwise all three stages use their normal
    cache TTLs (see each module).

    Columns: see `RESULT_COLUMNS` (quality_scorer's full per-ticker schema,
    plus timing_signal/timing_note/return_1y/rsi_14d/rsi_14d_direction/
    market_cap_b). `result.attrs["summary"]` carries
    {'scanned', 'dislocation', 'quality_pass'} counts for the report header.
    """
    if force_refresh:
        _clear_todays_cache()
        tickers = universe.fetch_russell_1000(force_refresh=True)["ticker"].tolist()
    else:
        tickers = universe.get_universe_tickers()

    price_df = price_filter.run_price_filter(tickers)

    rows = []
    for _, prow in price_df.iterrows():
        ticker = prow["ticker"]
        quality = quality_scorer.score_ticker(ticker, settings.FMP_API_KEY)

        if quality["quality_score"] < settings.MIN_QUALITY_SCORE or quality["red_flags"]:
            continue

        timing = timing_signal.classify_timing(prow["rsi_14d"], prow["rsi_14d_direction"])

        rows.append({
            **quality,
            **timing,
            "return_1y": prow["return_1y"],
            "rsi_14d": prow["rsi_14d"],
            "rsi_14d_direction": prow["rsi_14d_direction"],
            "market_cap_b": prow.get("market_cap_b"),
        })

    result = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    if not result.empty:
        signal_rank = result["timing_signal"].map(_SIGNAL_ORDER).fillna(len(_SIGNAL_ORDER))
        result = (
            result.assign(_signal_rank=signal_rank)
            .sort_values(["_signal_rank", "quality_score"], ascending=[True, False])
            .drop(columns="_signal_rank")
            .head(settings.MAX_SCREENER_OUTPUT)
            .reset_index(drop=True)
        )

    summary = {"scanned": len(tickers), "dislocation": len(price_df), "quality_pass": len(result)}
    result.attrs["summary"] = summary
    logger.info(
        "Screener: scanned=%d dislocation=%d quality_pass=%d",
        summary["scanned"], summary["dislocation"], summary["quality_pass"],
    )
    return result


def _format_row(rank: int, row: pd.Series) -> str:
    arrow = _RSI_ARROWS.get(row["rsi_14d_direction"], "→")
    roic = row["roic_current"]
    roic_str = f"{roic:.0f}%" if roic is not None and pd.notna(roic) else "n/a"
    net_debt_ebitda = row["net_debt_ebitda"]
    leverage_str = f"{net_debt_ebitda:.1f}x" if net_debt_ebitda is not None and pd.notna(net_debt_ebitda) else "n/a"
    margin = row["ebitda_margin_3y_avg"]
    margin_str = f"{margin:.0f}%" if margin is not None and pd.notna(margin) else "n/a"
    rev_cagr = row["revenue_cagr_3y"]
    rev_str = f"{rev_cagr:+.0f}%" if rev_cagr is not None and pd.notna(rev_cagr) else "n/a"
    sector = row["sector"] or "n/a"

    line1 = (
        f"{rank}. {row['ticker']:<4} Score:{row['quality_score']:.0f} | ROIC:{roic_str} | "
        f"{row['return_1y']:+.0f}% 1Y | RSI14:{row['rsi_14d']:.0f}{arrow} | Rev{rev_str}"
    )
    line2 = f"   Sector: {sector} | Net Debt: {leverage_str} | Margin: {margin_str}"
    return line1 + "\n" + line2


def format_screener_output(df: pd.DataFrame) -> str:
    """Render `run_screener()`'s output DataFrame as a Telegram-ready string."""
    summary = df.attrs.get("summary", {})
    scanned = summary.get("scanned", "?")
    dislocation = summary.get("dislocation", "?")
    quality_pass = summary.get("quality_pass", len(df))

    lines = [
        f"🔍 SCREENER — {date.today().isoformat()}",
        _DIVIDER,
        f"Scanned: {scanned} | Dislocation: {dislocation} | Quality pass: {quality_pass}",
    ]

    if df.empty:
        lines.append("")
        lines.append("No names cleared the quality bar today.")
    else:
        current_signal = None
        for i, (_, row) in enumerate(df.iterrows(), start=1):
            if row["timing_signal"] != current_signal:
                current_signal = row["timing_signal"]
                lines.append("")
                lines.append(_SIGNAL_HEADERS.get(current_signal, current_signal))
            lines.append(_format_row(i, row))

    lines.append(_DIVIDER)
    lines.append("/discuss TICKER for deep dive")
    return "\n".join(lines)


def debug_ticker(ticker: str) -> dict:
    """Run `quality_scorer.score_ticker()` on a single name and print the full result.

    Bypasses the price-filter/universe stages entirely — useful for checking
    why a specific ticker scores the way it does, or for inspecting
    `data_warnings` when a fetch looks off. Returns the result dict.
    """
    ticker = ticker.upper()
    result = quality_scorer.score_ticker(ticker, settings.FMP_API_KEY)

    print(f"\n{'=' * 60}")
    print(f"DEBUG: {ticker}")
    print(f"{'=' * 60}")
    print(f"quality_score: {result['quality_score']}  |  tier: {result['tier']}")

    print("\n--- score_components ---")
    pprint(result["score_components"])

    print("\n--- red_flags ---")
    pprint(result["red_flags"])

    print("\n--- yellow_flags ---")
    pprint(result["yellow_flags"])

    print("\n--- data_warnings ---")
    pprint(result["data_warnings"])

    print("\n--- full result ---")
    pprint(result)
    print()

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1:
        debug_ticker(sys.argv[1])
    else:
        output_df = run_screener()
        print(format_screener_output(output_df))
