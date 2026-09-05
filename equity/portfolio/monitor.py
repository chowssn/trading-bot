"""Daily portfolio price monitor.

Pulls current price action for every name in
`equity.config.positions.POSITIONS`, plus a lightweight check on
`WATCHLIST` names not yet in a position.

Size, cost basis, and market value come from `equity.config.positions` —
`POSITIONS`/`WATCHLIST` there are already merged with
`config/positions_override.json` at import time (see
`positions._apply_overrides()`), so every ticker's config dict here
(whether reached via `POSITIONS.items()`/`WATCHLIST.items()` or
`positions_config.get_position()`) carries the IBKR-imported `avg_cost`,
`size_pct`, and `market_value` fields when available. A ticker only in
the override file (no thesis entry in `positions.py`) still shows up here
because the override merge adds it to `POSITIONS`/`WATCHLIST` directly —
there is no separate override lookup needed in this module. Tickers with
no override data yet (`size_pct` still `None`) render "Size: pending
IBKR".

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
from equity.config.market_config import POSITION_SIZE_ALERT_ENABLED, POSITION_TIERS

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


def _build_entry(ticker: str, config: dict) -> dict | None:
    """Build one ticker's price-action entry, or None if price data couldn't be fetched.

    `config` is expected to already be the merged dict for `ticker` — i.e.
    a value from `positions_config.POSITIONS`/`WATCHLIST` (or
    `positions_config.get_position(ticker)`), which folds in
    `positions_override.json` fields (`avg_cost`, `size_pct`,
    `market_value`) at import time. This function doesn't re-read the
    override file itself.
    """
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
        "avg_cost": config.get("avg_cost"),
        "size_pct": config.get("size_pct"),
        "market_value": config.get("market_value"),
        "move_flag": _move_flag(change_1d_pct),
        "thesis_status": "THESIS_BREAKING" if config.get("stop_thesis") else "OK",
        "tier": config.get("tier", ""),
        "tier_v2": config.get("tier_v2", ""),
        "sector": config.get("sector", ""),
    }


def run_portfolio_monitor() -> dict:
    """Fetch price action for every ticker in `positions.POSITIONS`.

    Individual ticker fetch failures are logged, surfaced as an alert, and
    skipped — never crash the whole run over one bad ticker. WATCHLIST
    names are not visited here; `format_portfolio_monitor()` fetches them
    separately for the watchlist section.
    """
    result_positions = {}
    alerts = []

    for ticker, config in positions_config.POSITIONS.items():
        entry = _build_entry(ticker, config)
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


def _size_alert_flag(size_pct: float | None, tier_config: dict) -> str:
    """Flag when `size_pct` falls outside tier_config's min/max bounds, or '' if in bounds."""
    max_size = tier_config.get("max_size_pct", 0)
    min_size = tier_config.get("min_size_pct", 0)
    if not (POSITION_SIZE_ALERT_ENABLED and size_pct and max_size):
        return ""
    if size_pct > max_size:
        return f" ⚠️ oversized (>{max_size}% target)"
    if min_size and size_pct < min_size:
        return f" ⚠️ undersized (<{min_size}% target)"
    return ""


def _format_position_line(ticker: str, entry: dict) -> str:
    avg_cost = entry["avg_cost"]
    size_pct = entry["size_pct"]
    price_current = entry["price_current"]

    # tier_v2 (POSITION_TIERS) supersedes the legacy `tier` string for
    # display/sizing when a position has been classified under the new
    # framework; positions not yet reclassified (tier_v2 unset/unknown)
    # fall back to the old tier label with no size bounds to check.
    tier_config = POSITION_TIERS.get(entry.get("tier_v2", ""), {})
    tier_label = tier_config.get("label") or _tier_label(entry["tier"])
    size_flag = _size_alert_flag(size_pct, tier_config)

    if avg_cost and avg_cost > 0 and price_current:
        unrealized_pct = (price_current / avg_cost - 1) * 100
        size_str = f"{size_pct:.1f}% | Cost ${avg_cost:.2f} | P&L {unrealized_pct:+.1f}%"
    elif size_pct is not None:
        size_str = f"{size_pct:.1f}%"
    else:
        size_str = "Size: pending IBKR"

    return (
        f"{ticker:<5} {entry['change_1d_pct']:+.1f}%  ${price_current:.0f} | "
        f"{size_str} | {tier_label}{size_flag}"
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
            entry = _build_entry(ticker, config)
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
