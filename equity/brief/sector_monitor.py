"""Sector/concentration risk monitor: position correlations, sector breakdown, options IV.

`fetch_sector_data()` is one batched `yf.download()` over
`market_config.BENCHMARK_TICKERS` + every `positions.POSITIONS` ticker
(mirrors `equity.brief.market_snapshot`'s batching approach), cached 1
hour to `equity/data/cache/sector_monitor.json` — a cache written on a
prior calendar date is always treated as stale regardless of CACHE_HOURS,
so an 11pm write never serves yesterday's prices the next morning. The benchmark tickers
ride along in that same batch per the spec this module was built from,
but aren't consumed by anything in this module yet — `POSITION_SECTOR_MAP`
import is kept for parity with `performance_tracker.py` and future
position-vs-sector-ETF correlation work, not used directly here either.

Four independent pieces of data, each degrading gracefully on its own
rather than taking down the whole fetch:

- `compute_correlation_matrix()` / `find_concentration_flags()` — pairwise
  60-day-return correlation across `positions.POSITIONS` tickers only
  (never benchmarks). Only the *flags* (>0.70 concentrated, <-0.30 hedge)
  are returned for the brief — the full N×N matrix is deliberately not
  part of this module's output; see the module's `/discuss PORTFOLIO`
  note in `format_sector_section()`.
- `compute_sector_concentration()` — count-of-positions breakdown by each
  position's `sector` field (not dollar-weighted — position sizes aren't
  available until IBKR/Module 4 is connected, same caveat as
  `equity.portfolio.monitor`).
- `fetch_etf_flows_stub()` — IBKR stub, same shape/pattern as
  `equity.portfolio.monitor.get_ibkr_positions()`.
- `fetch_options_activity()` — front-month options chain per position
  ticker: ATM-average implied vol as an "iv30" proxy (there's no separate
  30-day-constant-maturity fetch here), put/call volume ratio, and a
  `skew_signal` (`PUT_HEAVY`/`CALL_HEAVY`/`NORMAL`) describing whether that
  ratio is unusual enough to trust.

  "Front-month" here means the nearest *standard monthly* (3rd-Friday)
  expiration (`_select_front_month()`), not simply the closest date —
  many names also list weekly expirations, and weeklies carry far less
  accumulated open interest, which makes volume/OI comparisons noisy.
  Confirmed empirically 2026-08-30: picking the literal nearest expiration
  (a weekly) made every one of CCJ/MSFT/UMAC/PGR/CEG look unusually active
  that day; switching to each name's nearest monthly helped but didn't fix
  it — the original rule flagged a chain as unusual if *any single strike*
  had volume > 2x its open interest, and with dozens of strikes on a chain
  a lone thin far-OTM strike (volume=1, open interest=0) is enough to trip
  that. `skew_signal` replaces it with a whole-chain approach instead of a
  single-strike one: `_classify_skew()` only trusts `put_call_ratio` once
  the *entire* chain clears two liquidity gates — total volume (calls +
  puts) exceeding `MIN_TOTAL_VOLUME` contracts (filters out illiquid names
  like UMAC, where a handful of contracts can swing the ratio to an
  extreme), and total volume exceeding `VOLUME_OI_RATIO`x total open
  interest (filters out a chain that's merely being priced/quoted rather
  than actively traded). A name that fails either gate reports `NORMAL`
  regardless of how extreme its `put_call_ratio` looks — see
  `_format_options_line()`'s CEG case, where a 1.6 P/C ratio still renders
  as `NORMAL` because the chain's volume doesn't clear the gates.

"Newly crossed 0.70 threshold" highlights compare this run's concentrated
pairs against whatever the *last* run recorded, persisted in
`equity/data/cache/concentration_history.json` — there's no scheduled
weekly cadence enforced anywhere in this codebase yet, so this is really
"since the last time this ran," not a strict calendar week. Documented as
such in `_build_highlights()` rather than overclaiming "vs last week."
"""

import json
import logging
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

from equity.config import positions as positions_config
from equity.config.market_config import (  # noqa: F401 — BENCHMARK_TICKERS/POSITION_SECTOR_MAP: see module docstring
    BENCHMARK_TICKERS,
    CORRELATION_CONCENTRATION_THRESHOLD,
    CORRELATION_HEDGE_THRESHOLD,
    POSITION_SECTOR_MAP,
    SECTOR_CONCENTRATION_MAX_PCT,
)

logger = logging.getLogger(__name__)

_DIVIDER = "━━━━━━━━━━━━━━━━━━━━━━━━"

_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache"
_CACHE_PATH = _CACHE_DIR / "sector_monitor.json"
CACHE_HOURS = 1

_CONCENTRATION_HISTORY_PATH = _CACHE_DIR / "concentration_history.json"

# Sector-breakdown flag threshold — position-count-based (see module
# docstring). Not in market_config: it's a lower-tier "worth a look" flag
# in "Your Sector Exposure" distinct from SECTOR_CONCENTRATION_MAX_PCT,
# which is the stricter threshold promoted to HIGHLIGHTS.
SECTOR_CONCENTRATION_FLAG_PCT = 30.0

# Options-activity display thresholds — local to this module, not sourced
# from market_config (mirrors equity.portfolio.monitor's local
# LARGE_MOVE_PCT/MOVE_PCT convention).
ELEVATED_IV30_PCT = 50.0

# skew_signal gates (see module docstring / _classify_skew()): a chain's
# put/call ratio is only trusted once the whole chain — not a single
# strike — clears both a minimum absolute volume and a minimum volume
# relative to open interest.
MIN_TOTAL_VOLUME = 5000
VOLUME_OI_RATIO = 1.5
PUT_HEAVY_PCR = 1.5
CALL_HEAVY_PCR = 0.4


# ---------------------------------------------------------------------------
# yfinance batch fetch
# ---------------------------------------------------------------------------

def _download_batch(tickers: list[str], period: str) -> pd.DataFrame | None:
    if not tickers:
        return None
    try:
        data = yf.download(
            tickers,
            period=period,
            auto_adjust=True,
            group_by="ticker",
            threads=True,
            progress=False,
        )
    except Exception as exc:  # yfinance can raise a variety of things on network/format issues
        logger.warning("yf.download failed for batch (period=%s): %s", period, exc)
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


def _build_close_frame(data: pd.DataFrame | None, tickers: list[str]) -> pd.DataFrame:
    """A ticker-keyed DataFrame of close-price series, dates auto-aligned. Missing tickers just don't appear."""
    closes = {}
    for ticker in tickers:
        close = _ticker_close(data, ticker)
        if close is not None:
            closes[ticker] = close
    return pd.DataFrame(closes)


def _safe_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Correlation matrix / concentration flags
# ---------------------------------------------------------------------------

def compute_correlation_matrix(price_data: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    returns = price_data[tickers].pct_change().dropna()
    return returns.corr()


def find_concentration_flags(corr_matrix: pd.DataFrame, threshold: float = CORRELATION_CONCENTRATION_THRESHOLD) -> list[dict]:
    """Every pair in `corr_matrix` above `threshold` ('concentrated') or below `CORRELATION_HEDGE_THRESHOLD` ('hedge').

    Only the upper triangle is walked — each unordered pair appears once.
    """
    flags = []
    tickers = list(corr_matrix.columns)
    for i, ticker1 in enumerate(tickers):
        for ticker2 in tickers[i + 1:]:
            corr = corr_matrix.loc[ticker1, ticker2]
            if pd.isna(corr):
                continue
            corr = float(corr)
            if corr > threshold:
                flags.append({"ticker1": ticker1, "ticker2": ticker2, "correlation": corr, "flag_type": "concentrated"})
            elif corr < CORRELATION_HEDGE_THRESHOLD:
                flags.append({"ticker1": ticker1, "ticker2": ticker2, "correlation": corr, "flag_type": "hedge"})
    return flags


def _load_previous_concentrated_pairs() -> set[frozenset] | None:
    if not _CONCENTRATION_HISTORY_PATH.exists():
        return None
    try:
        with open(_CONCENTRATION_HISTORY_PATH) as f:
            history = json.load(f)
        pairs = history["pairs"]
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        logger.warning("concentration_history.json unreadable, ignoring: %s", exc)
        return None
    return {frozenset(pair) for pair in pairs}


def _write_concentration_history(pairs: list[tuple[str, str]]) -> None:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(_CONCENTRATION_HISTORY_PATH, "w") as f:
            json.dump({"fetched_at": datetime.now().isoformat(), "pairs": [list(p) for p in pairs]}, f, indent=2)
    except OSError as exc:
        logger.warning("Failed to write concentration_history.json: %s", exc)


# ---------------------------------------------------------------------------
# Sector concentration
# ---------------------------------------------------------------------------

def compute_sector_concentration(positions: dict) -> dict:
    """Position-count sector breakdown of `positions` (a `positions.POSITIONS`-shaped dict).

    Returns more than the bare {sector: count} this was scoped as —
    `format_sector_section()` also needs the ticker lists and percentages
    to render "Your Sector Exposure" and to drive the >30%/>40% flags, so
    those are included too rather than recomputed twice.
    """
    tickers_by_sector: dict[str, list[str]] = {}
    for ticker, config in positions.items():
        sector = config.get("sector") or "Unknown"
        tickers_by_sector.setdefault(sector, []).append(ticker)

    total = len(positions)
    pct_breakdown = {
        sector: (len(tickers) / total * 100) if total else 0.0
        for sector, tickers in tickers_by_sector.items()
    }
    concentrated_sectors = [s for s, pct in pct_breakdown.items() if pct > SECTOR_CONCENTRATION_FLAG_PCT]

    return {
        "breakdown": {sector: len(tickers) for sector, tickers in tickers_by_sector.items()},
        "tickers_by_sector": tickers_by_sector,
        "pct_breakdown": pct_breakdown,
        "concentrated_sectors": concentrated_sectors,
        "total_positions": total,
    }


# ---------------------------------------------------------------------------
# ETF flows (stub) / options activity
# ---------------------------------------------------------------------------

def fetch_etf_flows_stub() -> dict:
    """IBKR ETF-flow stub (Module 4 — not yet connected). Same pattern as `monitor.get_ibkr_positions()`."""
    return {
        "ibkr_connected": False,
        "message": "ETF flow monitoring via IBKR available in Module 4",
    }


def _is_standard_monthly(expiry: date) -> bool:
    """True if `expiry` is the 3rd Friday of its month — the standard monthly options expiration."""
    return expiry.weekday() == 4 and 15 <= expiry.day <= 21


def _select_front_month(expirations: tuple[str, ...]) -> str | None:
    """The nearest standard-monthly expiration, or the nearest expiration at all if none qualify.

    See module docstring for why "nearest monthly" beats "nearest date"
    here. None if `expirations` is empty or none of it parses as a date.
    """
    parsed = []
    for exp_str in expirations:
        try:
            parsed.append(date.fromisoformat(exp_str))
        except ValueError:
            continue
    if not parsed:
        return None

    monthly = sorted(d for d in parsed if _is_standard_monthly(d))
    return (monthly[0] if monthly else min(parsed)).isoformat()


def _column_sum(chain_side: pd.DataFrame, column: str) -> float:
    return float(chain_side[column].fillna(0).sum()) if column in chain_side.columns else 0.0


def _chain_totals(calls: pd.DataFrame, puts: pd.DataFrame) -> tuple[float, float]:
    """(total_volume, total_open_interest) summed across both calls and puts."""
    total_volume = _column_sum(calls, "volume") + _column_sum(puts, "volume")
    total_oi = _column_sum(calls, "openInterest") + _column_sum(puts, "openInterest")
    return total_volume, total_oi


def _classify_skew(total_volume: float, total_oi: float, put_call_ratio: float | None) -> str:
    """'PUT_HEAVY' / 'CALL_HEAVY' / 'NORMAL' — see module docstring for the two liquidity gates.

    `put_call_ratio` is only trusted as a skew signal once the whole chain
    clears both: total volume > MIN_TOTAL_VOLUME contracts, and total
    volume > VOLUME_OI_RATIOx total open interest. Either gate failing
    (or no ratio to begin with) means 'NORMAL', regardless of how extreme
    the ratio looks — a thin chain can produce an extreme ratio from a
    handful of contracts that means nothing.
    """
    if put_call_ratio is None:
        return "NORMAL"
    if total_volume <= MIN_TOTAL_VOLUME or total_volume <= VOLUME_OI_RATIO * total_oi:
        return "NORMAL"
    if put_call_ratio > PUT_HEAVY_PCR:
        return "PUT_HEAVY"
    if put_call_ratio < CALL_HEAVY_PCR:
        return "CALL_HEAVY"
    return "NORMAL"


def fetch_options_activity(tickers: list[str]) -> dict:
    """Front-month options snapshot per ticker: {iv30, put_call_ratio, skew_signal}.

    Never raises: a ticker with no options chain, or any fetch failure,
    still gets an entry (iv30/put_call_ratio None, skew_signal 'NORMAL')
    rather than being dropped from the result.
    """
    result: dict[str, dict] = {}

    for ticker in tickers:
        entry = {"iv30": None, "put_call_ratio": None, "skew_signal": "NORMAL"}

        try:
            t = yf.Ticker(ticker)
            expirations = t.options
        except Exception as exc:  # yfinance can raise a variety of things on network/format issues
            logger.warning("Options expirations fetch failed for %s: %s", ticker, exc)
            result[ticker] = entry
            continue

        front_month = _select_front_month(expirations)
        if front_month is None:
            result[ticker] = entry
            continue

        try:
            chain = t.option_chain(front_month)
            current_price = t.fast_info.get("lastPrice")
        except Exception as exc:
            logger.warning("Options chain fetch failed for %s (%s): %s", ticker, front_month, exc)
            result[ticker] = entry
            continue

        calls, puts = chain.calls, chain.puts

        if not calls.empty and current_price:
            ivs = [_safe_float(calls.loc[(calls["strike"] - current_price).abs().idxmin()].get("impliedVolatility"))]
            if not puts.empty:
                ivs.append(_safe_float(puts.loc[(puts["strike"] - current_price).abs().idxmin()].get("impliedVolatility")))
            ivs = [v for v in ivs if v is not None]
            if ivs:
                entry["iv30"] = sum(ivs) / len(ivs) * 100

        call_volume = _safe_float(calls["volume"].sum()) if "volume" in calls.columns else None
        put_volume = _safe_float(puts["volume"].sum()) if "volume" in puts.columns else None
        if call_volume:
            entry["put_call_ratio"] = put_volume / call_volume if put_volume is not None else None

        total_volume, total_oi = _chain_totals(calls, puts)
        entry["skew_signal"] = _classify_skew(total_volume, total_oi, entry["put_call_ratio"])

        result[ticker] = entry

    return result


# ---------------------------------------------------------------------------
# Cache (1 hour)
# ---------------------------------------------------------------------------

def _load_cache() -> dict | None:
    if not _CACHE_PATH.exists():
        return None
    try:
        with open(_CACHE_PATH) as f:
            cache = json.load(f)
        fetched_at = datetime.fromisoformat(cache["fetched_at"])
        cached_date = cache.get("date")
        payload = cache["data"]
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.warning("sector_monitor.json cache unreadable, ignoring: %s", exc)
        return None

    if cached_date != date.today().isoformat():
        return None  # written on a prior trading day — force refresh regardless of CACHE_HOURS
    if (datetime.now() - fetched_at).total_seconds() > CACHE_HOURS * 3600:
        return None
    return payload


def _write_cache(data: dict) -> None:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(_CACHE_PATH, "w") as f:
            json.dump(
                {"fetched_at": datetime.now().isoformat(), "date": date.today().isoformat(), "data": data},
                f, indent=2,
            )
    except OSError as exc:
        logger.warning("Failed to write sector_monitor.json cache: %s", exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_sector_data() -> dict:
    """Correlation flags, sector concentration, options activity, and the ETF-flows stub. Cached 1 hour.

    Never raises: a failed batch download, a ticker with no usable price
    history, or a failed options fetch is recorded in `data_warnings`
    (and/or leaves that ticker's fields None) rather than crashing.
    """
    cached = _load_cache()
    if cached is not None:
        return cached

    data_warnings: list[str] = []
    position_tickers = list(positions_config.POSITIONS)
    all_tickers = sorted(set(position_tickers) | set(BENCHMARK_TICKERS))

    price_data = _download_batch(all_tickers, period="60d")
    if price_data is None:
        data_warnings.append("yf.download returned no data for the sector monitor batch")

    close_frame = _build_close_frame(price_data, position_tickers)
    missing = [t for t in position_tickers if t not in close_frame.columns]
    for ticker in missing:
        data_warnings.append(f"{ticker}: price data unavailable for correlation matrix")

    available_tickers = [t for t in position_tickers if t in close_frame.columns]
    if len(available_tickers) >= 2:
        corr_matrix = compute_correlation_matrix(close_frame, available_tickers)
        correlation_flags = find_concentration_flags(corr_matrix)
    else:
        data_warnings.append("Fewer than 2 positions with price history — correlation matrix skipped")
        correlation_flags = []

    current_pairs = [
        (flag["ticker1"], flag["ticker2"]) for flag in correlation_flags if flag["flag_type"] == "concentrated"
    ]
    previous_pairs = _load_previous_concentrated_pairs()
    if previous_pairs is None:
        newly_crossed_pairs = []
    else:
        newly_crossed_pairs = [pair for pair in current_pairs if frozenset(pair) not in previous_pairs]
    _write_concentration_history(current_pairs)

    sector_concentration = compute_sector_concentration(positions_config.POSITIONS)
    etf_flows = fetch_etf_flows_stub()
    options_activity = fetch_options_activity(position_tickers)

    result = {
        "correlation_flags": correlation_flags,
        "newly_crossed_pairs": newly_crossed_pairs,
        "sector_concentration": sector_concentration,
        "etf_flows": etf_flows,
        "options_activity": options_activity,
        "as_of": datetime.now().isoformat(),
        "data_warnings": data_warnings,
    }
    _write_cache(result)
    return result


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _format_exposure_line(sector: str, tickers: list[str], flagged: bool) -> str:
    count_word = "position" if len(tickers) == 1 else "positions"
    marker = " ⚠️" if flagged else ""
    return f"  {sector}: {', '.join(tickers)} — {len(tickers)} {count_word}{marker}"


def _format_correlation_line(flag: dict) -> str:
    if flag["flag_type"] == "concentrated":
        return f"  ⚠️ {flag['ticker1']} ↔ {flag['ticker2']}: {flag['correlation']:.2f} — concentrated risk"
    return f"  ✓ {flag['ticker1']} ↔ {flag['ticker2']}: {flag['correlation']:.2f} — natural hedge"


def _skew_label(skew: str) -> str:
    if skew == "PUT_HEAVY":
        return "PUT_HEAVY — unusual put buying (hedge/bearish)"
    if skew == "CALL_HEAVY":
        return "CALL_HEAVY — bullish positioning"
    return "NORMAL"


def _format_options_line(ticker: str, activity: dict) -> str:
    iv30 = activity.get("iv30")
    pcr = activity.get("put_call_ratio")
    skew = activity.get("skew_signal", "NORMAL")

    iv_str = f"IV30: {iv30:.0f}%" if iv30 is not None else "IV30: n/a"

    if pcr is None:
        # No call volume to build a ratio from (illiquid name, e.g. UMAC) —
        # a P/C ratio here would be meaningless, so fall back to IV alone.
        if iv30 is not None and iv30 > ELEVATED_IV30_PCT:
            return f"  {ticker:<5}  {iv_str} — elevated vol (speculative)"
        return f"  {ticker:<5}  {iv_str}"

    return f"  {ticker:<5}  {iv_str}  P/C: {pcr:.1f}  {_skew_label(skew)}"


def _build_highlights(sector_data: dict) -> list[str]:
    highlights = []

    for ticker1, ticker2 in sector_data.get("newly_crossed_pairs") or []:
        highlights.append(
            f"{ticker1} & {ticker2} newly crossed the {CORRELATION_CONCENTRATION_THRESHOLD:.2f} "
            f"correlation threshold since the last check"
        )

    pct_breakdown = (sector_data.get("sector_concentration") or {}).get("pct_breakdown", {})
    for sector, pct in pct_breakdown.items():
        if pct > SECTOR_CONCENTRATION_MAX_PCT:
            highlights.append(f"{sector} is {pct:.0f}% of your positions (by count) — concentration risk")

    for ticker, activity in (sector_data.get("options_activity") or {}).items():
        skew = activity.get("skew_signal")
        pcr = activity.get("put_call_ratio")
        if skew == "PUT_HEAVY":
            highlights.append(f"{ticker}: PUT_HEAVY skew (P/C {pcr:.1f}) — unusual put buying, potential hedge/bearish positioning")
        elif skew == "CALL_HEAVY":
            highlights.append(f"{ticker}: CALL_HEAVY skew (P/C {pcr:.1f}) — unusual call buying, potential bullish positioning")

    return highlights


def format_sector_section(sector_data: dict) -> str:
    """Render `fetch_sector_data()`'s output dict as a Telegram-ready string. Never raises."""
    lines = ["🏭 SECTOR & CONCENTRATION", _DIVIDER, "Your Sector Exposure"]

    sector_concentration = sector_data.get("sector_concentration", {})
    tickers_by_sector = sector_concentration.get("tickers_by_sector", {})
    concentrated_sectors = set(sector_concentration.get("concentrated_sectors", []))
    for sector, tickers in tickers_by_sector.items():
        lines.append(_format_exposure_line(sector, tickers, sector in concentrated_sectors))

    concentrated_flags = [f for f in sector_data.get("correlation_flags", []) if f["flag_type"] == "concentrated"]
    for flag in concentrated_flags:
        sector1 = positions_config.POSITIONS.get(flag["ticker1"], {}).get("sector", "?")
        sector2 = positions_config.POSITIONS.get(flag["ticker2"], {}).get("sector", "?")
        lines.append(
            f"  ⚠️ Concentration: {flag['ticker1']} ({sector1}) + {flag['ticker2']} ({sector2}) are "
            f"{flag['correlation'] * 100:.0f}% correlated"
        )
    lines.append("")

    lines.append("60-Day Correlations (flags only)")
    flags = sector_data.get("correlation_flags", [])
    if flags:
        lines.extend(_format_correlation_line(flag) for flag in flags)
    else:
        lines.append("  No correlation flags — all position pairs within -0.30 to 0.70")
    lines.append("  Full N×N matrix available via /discuss PORTFOLIO")
    lines.append("")

    lines.append("Options Activity (IV snapshot)")
    options_activity = sector_data.get("options_activity", {})
    if options_activity:
        lines.extend(_format_options_line(ticker, activity) for ticker, activity in options_activity.items())
    else:
        lines.append("  No options data available")
    lines.append("")

    etf_flows = sector_data.get("etf_flows", {})
    lines.append(f"ETF Flows: {etf_flows.get('message', 'Connect IBKR for institutional flow data')}")

    highlights = _build_highlights(sector_data)
    if highlights:
        lines.append("")
        lines.append("HIGHLIGHTS")
        lines.extend(f"• {h}" for h in highlights)

    warnings = sector_data.get("data_warnings") or []
    if warnings:
        lines.append("")
        lines.extend(f"⚠️ {w}" for w in warnings)

    lines.append(_DIVIDER)
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    sector_result = fetch_sector_data()
    print(format_sector_section(sector_result))
