"""Main orchestrator for the equity screener funnel.

Ties the three funnel stages together:

    universe.get_universe_tickers()   Russell 1000, from the IWB holdings CSV
        -> price_filter.run_price_filter()   cheap yfinance price/RSI/volume
           dislocation + market-cap pre-filter, ~1000 tickers -> ~30-80
           (this stage now also computes timing_signal/timing_note itself,
           via timing_signal.classify_timing() with the full RSI/volume/
           price series still in scope — see price_filter.py)
              -> quality_scorer.score_ticker()   FMP/yfinance fundamentals,
                 0-90 quality score, per surviving ticker
                 -> _check_entry_correlation()   price-correlation gate
                    against existing positions, for names that pass quality

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
import yfinance as yf

from equity.config import settings
from equity.config.market_config import CORRELATION_ENTRY_THRESHOLD, CORRELATION_LOOKBACK_DAYS
from equity.config.positions import POSITIONS, get_position
from equity.screener import price_filter, quality_scorer, universe

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
    "timing_signal", "timing_note", "return_1y", "return_3m", "price_action_type",
    "rsi_14d", "rsi_14d_direction", "market_cap_b",
    "max_correlation", "correlated_with", "correlation_flag", "relationship_notes",
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


def _check_entry_correlation(
    candidate_ticker: str, position_tickers: list[str], lookback_days: int = CORRELATION_LOOKBACK_DAYS
) -> dict:
    """Price correlation between `candidate_ticker` and each existing position, plus relationship context.

    Never raises: a failed batch download (or fewer than 2 tickers to
    correlate) just comes back with `max_correlation=0.0`/no flag rather
    than propagating.

    Returns:
    {
        'max_correlation': float,
        'correlated_with': str | None,   # ticker with highest |correlation|
        'correlation_flag': bool,        # True if |max_correlation| > CORRELATION_ENTRY_THRESHOLD
        'relationship_notes': list[str], # e.g. ['same sector as CCJ', 'defined peer of ABT']
    }
    """
    empty = {"max_correlation": 0.0, "correlated_with": None, "correlation_flag": False, "relationship_notes": []}
    if not position_tickers:
        return empty

    all_tickers = [candidate_ticker] + position_tickers
    try:
        hist = yf.download(all_tickers, period=f"{lookback_days + 5}d", auto_adjust=True, progress=False)["Close"]
        returns = hist.pct_change().dropna()
        corr_matrix = returns.corr()

        max_corr = 0.0
        corr_with = None
        for pos in position_tickers:
            if pos in corr_matrix.columns and candidate_ticker in corr_matrix.index:
                c = corr_matrix.loc[candidate_ticker, pos]
                if pd.isna(c):
                    continue
                if abs(c) > abs(max_corr):
                    max_corr = float(c)
                    corr_with = pos
    except Exception as exc:  # yfinance can raise a variety of things on network/format issues
        logger.warning("Correlation gate fetch failed for %s: %s", candidate_ticker, exc)
        return empty

    # Sector and defined-peer relationship notes — independent of price
    # correlation, so these can surface a relationship even before it shows
    # up in the 60-day return correlation above.
    relationship_notes = []
    candidate_info = get_position(candidate_ticker)
    for pos_ticker, pos_data in POSITIONS.items():
        if candidate_info and pos_data.get("sector") and pos_data.get("sector") == candidate_info.get("sector"):
            relationship_notes.append(f"same sector as {pos_ticker}")
        peers = pos_data.get("peer_tickers", [])
        if candidate_ticker in peers:
            relationship_notes.append(f"defined peer of {pos_ticker}")

    flag = abs(max_corr) > CORRELATION_ENTRY_THRESHOLD
    return {
        "max_correlation": round(max_corr, 2),
        "correlated_with": corr_with,
        "correlation_flag": flag,
        "relationship_notes": relationship_notes,
    }


def run_screener(force_refresh: bool = False, regime_flags: list[str] | None = None) -> pd.DataFrame:
    """Run the full screener funnel and return the ranked results DataFrame.

    If `force_refresh` is True, today's price-filter/quality-scorer cache
    files are cleared and the Russell 1000 universe is force-refreshed
    before the funnel runs — otherwise all three stages use their normal
    cache TTLs (see each module). `regime_flags` (today's active flags from
    `market_snapshot.fetch_market_snapshot()`) is passed through to
    `price_filter.run_price_filter()`, which uses it to set a
    regime-adjusted RSI dislocation threshold.

    Columns: see `RESULT_COLUMNS` (quality_scorer's full per-ticker schema,
    plus timing_signal/timing_note/return_1y/return_3m/price_action_type/
    rsi_14d/rsi_14d_direction/market_cap_b, plus the correlation-gate
    fields). `result.attrs["summary"]` carries {'scanned', 'dislocation',
    'quality_pass'} counts for the report header.
    """
    if force_refresh:
        _clear_todays_cache()
        tickers = universe.fetch_russell_1000(force_refresh=True)["ticker"].tolist()
    else:
        tickers = universe.get_universe_tickers()

    price_df = price_filter.run_price_filter(tickers, regime_flags=regime_flags)
    position_tickers = list(POSITIONS)

    rows = []
    for _, prow in price_df.iterrows():
        ticker = prow["ticker"]
        quality = quality_scorer.score_ticker(ticker, settings.FMP_API_KEY)

        if quality["quality_score"] < settings.MIN_QUALITY_SCORE or quality["red_flags"]:
            continue

        correlation = _check_entry_correlation(ticker, position_tickers)

        rows.append({
            **quality,
            "timing_signal": prow.get("timing_signal"),
            "timing_note": prow.get("timing_note"),
            "return_1y": prow["return_1y"],
            "return_3m": prow.get("return_3m"),
            "price_action_type": prow.get("price_action_type"),
            "rsi_14d": prow["rsi_14d"],
            "rsi_14d_direction": prow["rsi_14d_direction"],
            "market_cap_b": prow.get("market_cap_b"),
            **correlation,
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
    lines = [line1, line2]

    price_action = row.get("price_action_type")
    return_3m = row.get("return_3m")
    if price_action and pd.notna(price_action):
        return_3m_str = f"{return_3m:+.0f}% 3M" if return_3m is not None and pd.notna(return_3m) else "3M n/a"
        lines.append(f"   Price action: {price_action} ({return_3m_str})")

    fwd_pe = row.get("forward_pe")
    if fwd_pe is not None and pd.notna(fwd_pe):
        lines.append(f"   Fwd P/E: {fwd_pe:.1f}x (display only — verify against Bloomberg)")

    if row.get("correlation_flag"):
        lines[0] += f"  ⚠️ {row['max_correlation']:.2f} corr with {row['correlated_with']}"
    relationship_notes = row.get("relationship_notes") or []
    if relationship_notes:
        lines.append(f"   Rel: {'; '.join(relationship_notes)}")

    return "\n".join(lines)


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
