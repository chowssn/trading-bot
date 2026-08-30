"""Daily portfolio price monitor.

Pulls current price action for every name in
`equity.config.positions.POSITIONS`, plus a lightweight check on
`WATCHLIST` names not yet in a position.

No P&L or position sizing here — those depend on entry price and share
count, which come from IBKR (Module 4, not yet connected; see
`get_ibkr_positions()` below and the `equity.config.positions` module
docstring). Every `POSITIONS` ticker renders with live price action and
"Size: pending IBKR" until that connection lands.

Price data comes from `yf.Ticker(ticker).history(period='5d')` rather than
`.info` or a batched `yf.download()` — this module runs on a handful of
tickers at a time (not the ~1000-name screener universe), so per-ticker
call overhead doesn't matter, and `.history()` gives us both the latest
close and the prior close in one call without relying on `.info`'s
`previousClose` field (which yfinance sometimes returns stale/None for).
"""

import logging
from datetime import datetime

import yfinance as yf

from equity.config import positions as positions_config

logger = logging.getLogger(__name__)

_DIVIDER = "━━━━━━━━━━━━━━━━━━━━━━━━"

LARGE_MOVE_PCT = 3.0
MOVE_PCT = 1.0

_TIER_DISPLAY = {
    "core": "Core",
    "high_conviction": "High Conviction",
    "speculative": "Speculative 🚀",
    "watchlist": "Watchlist",
}


def _move_flag(change_1d_pct: float) -> str:
    if change_1d_pct > LARGE_MOVE_PCT:
        return "LARGE_UP"
    if change_1d_pct < -LARGE_MOVE_PCT:
        return "LARGE_DOWN"
    if change_1d_pct > MOVE_PCT:
        return "UP"
    if change_1d_pct < -MOVE_PCT:
        return "DOWN"
    return "FLAT"


def _fetch_price(ticker: str) -> tuple[float, float] | None:
    """Return (price_current, price_prev_close) from a 5-day history, or None on failure."""
    try:
        hist = yf.Ticker(ticker).history(period="5d")
    except Exception as exc:  # yfinance can raise a variety of things on network/format issues
        logger.warning("Price history fetch failed for %s: %s", ticker, exc)
        return None

    close = hist["Close"].dropna() if "Close" in hist.columns else hist.iloc[0:0]
    if len(close) < 2:
        logger.warning("Not enough price history for %s (%d row(s)) — need >= 2", ticker, len(close))
        return None

    return float(close.iloc[-1]), float(close.iloc[-2])


def get_ibkr_positions() -> dict:
    """Stub for live IBKR position sizes (Module 4 — not yet connected).

    Returns an empty dict for now. Once wired up this should return
    {ticker: size_pct} for every held position; callers treat a ticker
    missing from this dict as "size pending IBKR" rather than an error.
    """
    return {}


def _build_entry(ticker: str, config: dict, ibkr_positions: dict) -> dict | None:
    """Build one ticker's price-action entry, or None if price data couldn't be fetched."""
    prices = _fetch_price(ticker)
    if prices is None:
        return None
    price_current, price_prev_close = prices

    change_1d_abs = price_current - price_prev_close
    change_1d_pct = (price_current / price_prev_close - 1) * 100 if price_prev_close else 0.0

    return {
        "price_current": price_current,
        "change_1d_pct": change_1d_pct,
        "change_1d_abs": change_1d_abs,
        "size_pct": ibkr_positions.get(ticker),
        "move_flag": _move_flag(change_1d_pct),
        "thesis_status": "THESIS_BREAKING" if config.get("stop_thesis") else "OK",
        "tier": config.get("tier", ""),
        "sector": config.get("sector", ""),
    }


def run_portfolio_monitor() -> dict:
    """Fetch price action for every ticker in `positions.POSITIONS`.

    Individual ticker fetch failures are logged, surfaced as an alert, and
    skipped — never crash the whole run over one bad ticker. WATCHLIST
    names are not visited here; `format_portfolio_monitor()` fetches them
    separately for the watchlist section.
    """
    ibkr_positions = get_ibkr_positions()
    result_positions = {}
    alerts = []

    for ticker, config in positions_config.POSITIONS.items():
        entry = _build_entry(ticker, config, ibkr_positions)
        if entry is None:
            alerts.append(f"{ticker}  No price data available")
            continue

        result_positions[ticker] = entry

        if entry["move_flag"] in ("LARGE_UP", "LARGE_DOWN"):
            alerts.append(f"{ticker}  {entry['move_flag']}: {entry['change_1d_pct']:+.1f}%")
        if entry["thesis_status"] == "THESIS_BREAKING":
            alerts.append(f"{ticker}  THESIS_BREAKING — stop_thesis flagged in config")

    return {
        "as_of": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "positions": result_positions,
        "alerts": alerts,
    }


def _format_header_date(as_of: str) -> str:
    try:
        return datetime.strptime(as_of[:10], "%Y-%m-%d").strftime("%b %d, %Y")
    except ValueError:
        return as_of


def _tier_label(tier: str) -> str:
    return _TIER_DISPLAY.get(tier, tier.replace("_", " ").title())


def _format_position_line(ticker: str, entry: dict) -> str:
    size = entry["size_pct"]
    size_str = f"{size:.1f}%" if size is not None else "pending IBKR"
    return (
        f"{ticker:<5} {entry['change_1d_pct']:+.1f}%  ${entry['price_current']:.0f} | "
        f"Size: {size_str} | {_tier_label(entry['tier'])}"
    )


def _format_watchlist_line(ticker: str, entry: dict) -> str:
    return f"{ticker:<5} {entry['change_1d_pct']:+.1f}%  ${entry['price_current']:.0f}"


def format_portfolio_monitor(monitor_data: dict) -> str:
    """Render `run_portfolio_monitor()`'s output dict as a Telegram-ready string.

    The 👁 WATCHLIST section is fetched fresh here from `positions.WATCHLIST`
    rather than from `monitor_data`, since `run_portfolio_monitor()` only
    covers `POSITIONS`.
    """
    lines = [f"📁 PORTFOLIO — {_format_header_date(monitor_data['as_of'])}", _DIVIDER]

    alerts = monitor_data.get("alerts", [])
    if alerts:
        lines.append("⚠️ ALERTS")
        lines.extend(alerts)
        lines.append("")

    lines.append("📊 POSITIONS")
    active = monitor_data.get("positions", {})
    if active:
        lines.extend(_format_position_line(ticker, entry) for ticker, entry in active.items())
    else:
        lines.append("(no price data)")
    lines.append("")

    lines.append("👁 WATCHLIST")
    watchlist = positions_config.WATCHLIST
    if not watchlist:
        lines.append("(empty)")
    else:
        for ticker, config in watchlist.items():
            entry = _build_entry(ticker, config, {})
            if entry is None:
                lines.append(f"{ticker:<5} No price data available")
            else:
                lines.append(_format_watchlist_line(ticker, entry))

    lines.append(_DIVIDER)
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    monitor_result = run_portfolio_monitor()
    print(format_portfolio_monitor(monitor_result))
