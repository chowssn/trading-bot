"""Market-wide snapshot (equities, VIX, rates, commodities/FX, regime) for the morning brief.

One batched `yf.download()` call covers every ticker in
`market_config.EQUITY_BENCHMARKS`, `COMMODITY_TICKERS`, `RATE_TICKERS`, plus
`^VIX` — mirrors `equity.screener.price_filter`'s batching approach, just
for a handful of macro tickers instead of the Russell 1000.

Every threshold used for regime classification (`REGIME_RULES`,
`VIX_ELEVATED`/`VIX_HIGH`/`VIX_EXTREME`) is imported from
`equity.config.market_config` — nothing regime-related is hardcoded here.

Fed funds effective rate + target range come from FRED (`DFF`,
`DFEDTARU`, `DFEDTARL`) via `fredapi`, cached to
`equity/data/cache/fed_funds.json` for `FED_FUNDS_CACHE_HOURS` hours —
same cache-file shape (`fetched_at` + payload, age-checked on read) as
`eco_calendar.py`'s FOMC cache.
"""

import json
import logging
import operator
import os
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from fredapi import Fred

from equity.config.market_config import (
    COMMODITY_TICKERS,
    EQUITY_BENCHMARKS,
    RATE_TICKERS,
    REGIME_RULES,
    VIX_ELEVATED,
    VIX_EXTREME,
    VIX_HIGH,
)

load_dotenv()

logger = logging.getLogger(__name__)

FRED_API_KEY = os.getenv("FRED_API_KEY", "")

_DIVIDER = "━━━━━━━━━━━━━━━━━━━━━━━━"

_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache"
_FED_FUNDS_CACHE_PATH = _CACHE_DIR / "fed_funds.json"
FED_FUNDS_CACHE_HOURS = 24

VIX_TICKER = "^VIX"

# Display labels — presentation only, not thresholds, so not sourced from
# market_config (which has none for these).
_RATE_LABELS = {"^TNX": "10Y", "^IRX": "3M"}
_COMMODITY_FORMATS = {
    "DX-Y.NYB": {"label": "DXY", "prefix": "", "decimals": 1},
    "GC=F": {"label": "Gold", "prefix": "$", "decimals": 0},
    "CL=F": {"label": "Oil", "prefix": "$", "decimals": 1},
    "HG=F": {"label": "Copper", "prefix": "$", "decimals": 2},
}

# Maps each REGIME_RULES condition key to (regime_inputs key, comparator).
# REGIME_RULES itself — names, condition keys, and thresholds — all come
# from market_config; this table only says how to *evaluate* a condition
# key, not what triggers it.
_CONDITION_GETTERS = {
    "spy_change_1d_lt": ("spy_change_1d", operator.lt),
    "spy_change_1d_gt": ("spy_change_1d", operator.gt),
    "vix_gt": ("vix_current", operator.gt),
    "qqq_vs_spy_lt": ("qqq_vs_spy", operator.lt),
    "qqq_vs_spy_gt": ("qqq_vs_spy", operator.gt),
    "dxy_change_gt": ("dxy_change_1d", operator.gt),
    "dxy_change_lt": ("dxy_change_1d", operator.lt),
    "yield_10y_change_bps_gt": ("yield_10y_change_bps", operator.gt),
    "yield_10y_change_bps_lt": ("yield_10y_change_bps", operator.lt),
}


# ---------------------------------------------------------------------------
# yfinance batch fetch
# ---------------------------------------------------------------------------

def _download_batch(tickers: list[str]) -> pd.DataFrame | None:
    try:
        data = yf.download(
            tickers,
            period="ytd",
            auto_adjust=True,
            group_by="ticker",
            threads=True,
            progress=False,
        )
    except Exception as exc:  # yfinance can raise a variety of things on network/format issues
        logger.warning("yf.download failed for market snapshot batch: %s", exc)
        return None
    return None if data.empty else data


def _ticker_close(data: pd.DataFrame | None, ticker: str) -> pd.Series | None:
    """Close-price series for `ticker` from a multi-ticker yf.download() result, or None."""
    if data is None:
        return None
    try:
        if isinstance(data.columns, pd.MultiIndex):
            if ticker not in data.columns.get_level_values(0):
                return None
            sub = data[ticker]
        else:
            sub = data
        if "Close" not in sub.columns:
            return None
        close = sub["Close"].dropna()
        return close if not close.empty else None
    except Exception as exc:
        logger.warning("Could not extract close series for %s: %s", ticker, exc)
        return None


def _vix_regime(vix_current: float) -> str:
    """VIX tier label from `market_config.VIX_ELEVATED/VIX_HIGH/VIX_EXTREME`.

    Independent of `REGIME_RULES` (which only encodes ELEVATED_VOL/HIGH_VOL
    as regime_flags, with no EXTREME entry) — this covers the full ladder,
    including the extreme/crisis tier, as a field on the `vix` section.
    """
    if vix_current >= VIX_EXTREME:
        return "EXTREME"
    if vix_current >= VIX_HIGH:
        return "HIGH"
    if vix_current >= VIX_ELEVATED:
        return "ELEVATED"
    return "NORMAL"


# ---------------------------------------------------------------------------
# Fed funds rate (FRED, 24h cache)
# ---------------------------------------------------------------------------

def _load_fed_funds_cache() -> dict | None:
    if not _FED_FUNDS_CACHE_PATH.exists():
        return None
    try:
        with open(_FED_FUNDS_CACHE_PATH) as f:
            cache = json.load(f)
        fetched_at = datetime.fromisoformat(cache["fetched_at"])
        fed_funds = cache["fed_funds"]
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.warning("fed_funds.json cache unreadable, ignoring: %s", exc)
        return None

    if (datetime.now() - fetched_at).total_seconds() > FED_FUNDS_CACHE_HOURS * 3600:
        return None
    return fed_funds


def _write_fed_funds_cache(fed_funds: dict) -> None:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(_FED_FUNDS_CACHE_PATH, "w") as f:
            json.dump({"fetched_at": datetime.now().isoformat(), "fed_funds": fed_funds}, f, indent=2)
    except OSError as exc:
        logger.warning("Failed to write fed_funds.json cache: %s", exc)


def _fetch_fed_funds(data_warnings: list[str]) -> dict | None:
    """(effective_rate, target_upper, target_lower) from FRED, cache-first. None (+ warning) on failure."""
    cached = _load_fed_funds_cache()
    if cached is not None:
        return cached

    if not FRED_API_KEY:
        data_warnings.append("FRED_API_KEY not set — Fed funds rate unavailable")
        return None

    try:
        fred = Fred(api_key=FRED_API_KEY)
        effective_rate = float(fred.get_series("DFF").dropna().iloc[-1])
        target_upper = float(fred.get_series("DFEDTARU").dropna().iloc[-1])
        target_lower = float(fred.get_series("DFEDTARL").dropna().iloc[-1])
    except Exception as exc:
        logger.warning("FRED Fed funds fetch failed: %s", exc)
        data_warnings.append(f"Fed funds rate fetch failed: {exc}")
        return None

    fed_funds = {"effective_rate": effective_rate, "target_upper": target_upper, "target_lower": target_lower}
    _write_fed_funds_cache(fed_funds)
    return fed_funds


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------

def _apply_regime_rules(regime_inputs: dict) -> list[str]:
    """Evaluate every rule in `market_config.REGIME_RULES` against `regime_inputs`.

    A rule whose required input is unavailable (None) is silently skipped —
    missing data means "can't classify," not "flag triggered."
    """
    flags = []
    for name, condition in REGIME_RULES.items():
        if len(condition) != 1:
            logger.warning("Skipping malformed REGIME_RULES entry %r: expected exactly one condition key", name)
            continue
        condition_key, threshold = next(iter(condition.items()))
        getter = _CONDITION_GETTERS.get(condition_key)
        if getter is None:
            logger.warning("Skipping REGIME_RULES entry %r: unknown condition key %r", name, condition_key)
            continue
        input_key, comparator = getter
        value = regime_inputs.get(input_key)
        if value is None:
            continue
        if comparator(value, threshold):
            flags.append(name)
    return flags


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_market_snapshot() -> dict:
    """Batch-fetch equities/VIX/rates/commodities, classify the day's regime, and return one dict.

    Never raises: a failed batch download, a missing ticker, or a failed
    Fed funds fetch is recorded in `data_warnings` and the affected section
    is omitted/left partial rather than crashing the whole snapshot.
    """
    data_warnings: list[str] = []

    tickers = list(EQUITY_BENCHMARKS) + list(COMMODITY_TICKERS) + list(RATE_TICKERS) + [VIX_TICKER]
    data = _download_batch(tickers)
    if data is None:
        data_warnings.append("yf.download returned no data for the market snapshot batch")

    # --- Equities ---
    equities: dict[str, dict] = {}
    for ticker in EQUITY_BENCHMARKS:
        close = _ticker_close(data, ticker)
        if close is None or len(close) < 2:
            data_warnings.append(f"{ticker}: price data unavailable")
            continue
        price = float(close.iloc[-1])
        prev_close = float(close.iloc[-2])
        first_close = float(close.iloc[0])
        equities[ticker] = {
            "price": price,
            "change_1d_pct": (price / prev_close - 1) * 100,
            "change_ytd_pct": (price / first_close - 1) * 100 if first_close else None,
        }

    # --- VIX ---
    vix: dict = {}
    vix_close = _ticker_close(data, VIX_TICKER)
    if vix_close is None or len(vix_close) < 2:
        data_warnings.append(f"{VIX_TICKER}: price data unavailable")
    else:
        vix_current = float(vix_close.iloc[-1])
        rolling_20d = vix_close.rolling(20).mean().iloc[-1]
        vix = {
            "vix_current": vix_current,
            "vix_change_1d": vix_current - float(vix_close.iloc[-2]),
            "vix_20d_avg": float(rolling_20d) if pd.notna(rolling_20d) else None,
            "vix_regime": _vix_regime(vix_current),
        }
        if pd.isna(rolling_20d):
            data_warnings.append(f"{VIX_TICKER}: fewer than 20 trading days — vix_20d_avg unavailable")

    # --- Rates ---
    rates: dict[str, dict] = {}
    yield_10y = yield_10y_change_bps = yield_3m = None
    for ticker in RATE_TICKERS:
        close = _ticker_close(data, ticker)
        if close is None or len(close) < 2:
            data_warnings.append(f"{ticker}: price data unavailable")
            continue
        yield_pct = float(close.iloc[-1])
        change_1d_bps = (yield_pct - float(close.iloc[-2])) * 100
        rates[ticker] = {"yield_pct": yield_pct, "change_1d_bps": change_1d_bps}
        if ticker == "^TNX":
            yield_10y, yield_10y_change_bps = yield_pct, change_1d_bps
        elif ticker == "^IRX":
            yield_3m = yield_pct

    yield_curve_2s10s_approx = None
    if yield_10y is not None and yield_3m is not None:
        yield_curve_2s10s_approx = yield_10y - yield_3m  # 3M used as 2Y proxy — see module docstring

    # --- Commodities & FX ---
    commodities: dict[str, dict] = {}
    dxy_change_1d = None
    for ticker in COMMODITY_TICKERS:
        close = _ticker_close(data, ticker)
        if close is None or len(close) < 2:
            data_warnings.append(f"{ticker}: price data unavailable")
            continue
        price = float(close.iloc[-1])
        change_1d_pct = (price / float(close.iloc[-2]) - 1) * 100
        commodities[ticker] = {"price": price, "change_1d_pct": change_1d_pct}
        if ticker == "DX-Y.NYB":
            dxy_change_1d = change_1d_pct

    # --- Fed funds ---
    fed_funds = _fetch_fed_funds(data_warnings)

    # --- Regime classification ---
    spy = equities.get("SPY")
    qqq = equities.get("QQQ")
    regime_inputs = {
        "spy_change_1d": spy["change_1d_pct"] if spy else None,
        "vix_current": vix.get("vix_current"),
        "qqq_vs_spy": (qqq["change_1d_pct"] - spy["change_1d_pct"]) if spy and qqq else None,
        "dxy_change_1d": dxy_change_1d,
        "yield_10y_change_bps": yield_10y_change_bps,
    }
    regime_flags = _apply_regime_rules(regime_inputs)

    return {
        "equities": equities,
        "vix": vix,
        "rates": rates,
        "yield_curve_2s10s_approx": yield_curve_2s10s_approx,
        "commodities": commodities,
        "fed_funds": fed_funds,
        "regime_flags": regime_flags,
        "as_of": datetime.now().isoformat(),
        "data_warnings": data_warnings,
    }


def _fmt_amount(value: float, prefix: str, decimals: int) -> str:
    return f"{prefix}{value:,.{decimals}f}"


def format_market_snapshot(snapshot: dict) -> str:
    """Render `fetch_market_snapshot()`'s output dict as a Telegram-ready string."""
    try:
        header_date = datetime.fromisoformat(snapshot["as_of"]).strftime("%b %d, %Y")
    except (KeyError, TypeError, ValueError):
        header_date = date.today().strftime("%b %d, %Y")

    lines = [f"📊 MARKET SNAPSHOT — {header_date}", _DIVIDER, "Equities"]

    equities = snapshot.get("equities", {})
    for ticker in EQUITY_BENCHMARKS:
        eq = equities.get(ticker)
        if eq is None:
            lines.append(f"  {ticker:<5} data unavailable")
            continue
        ytd = eq["change_ytd_pct"]
        ytd_str = f"YTD{ytd:+.1f}%" if ytd is not None else "YTD n/a"
        lines.append(f"  {ticker:<5} ${eq['price']:,.0f}  {eq['change_1d_pct']:+.1f}%  {ytd_str}")

    vix = snapshot.get("vix") or {}
    if vix:
        avg = vix.get("vix_20d_avg")
        avg_str = f"  20d avg: {avg:.1f}" if avg is not None else ""
        lines.append(f"  VIX   {vix['vix_current']:.1f}  ({vix['vix_change_1d']:+.1f}){avg_str}")
    else:
        lines.append("  VIX   data unavailable")

    lines.append("")
    lines.append("Rates")
    rates = snapshot.get("rates", {})
    for ticker, label in _RATE_LABELS.items():
        r = rates.get(ticker)
        if r is None:
            lines.append(f"  {label:<5} data unavailable")
            continue
        lines.append(f"  {label:<5} {r['yield_pct']:.2f}%  ({r['change_1d_bps']:+.0f}bp)")

    curve = snapshot.get("yield_curve_2s10s_approx")
    curve_str = f"{curve * 100:+.0f}bp (3M proxy)" if curve is not None else "data unavailable"
    lines.append(f"  Curve  {curve_str}")

    fed_funds = snapshot.get("fed_funds")
    fed_str = f"{fed_funds['target_lower']:.2f}-{fed_funds['target_upper']:.2f}%" if fed_funds else "data unavailable"
    lines.append(f"  Fed:  {fed_str}")

    lines.append("")
    lines.append("Commodities & FX")
    commodities = snapshot.get("commodities", {})
    for ticker, fmt in _COMMODITY_FORMATS.items():
        c = commodities.get(ticker)
        if c is None:
            lines.append(f"  {fmt['label']:<6} data unavailable")
            continue
        price_str = _fmt_amount(c["price"], fmt["prefix"], fmt["decimals"])
        lines.append(f"  {fmt['label']:<6} {price_str}  ({c['change_1d_pct']:+.1f}%)")

    regime_flags = snapshot.get("regime_flags") or []
    if regime_flags:
        lines.append("")
        lines.append(f"Regime: {' | '.join(regime_flags)}")

    warnings = snapshot.get("data_warnings") or []
    if warnings:
        lines.append("")
        lines.extend(f"⚠️ {w}" for w in warnings)

    lines.append(_DIVIDER)
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    snapshot_result = fetch_market_snapshot()
    print(format_market_snapshot(snapshot_result))
