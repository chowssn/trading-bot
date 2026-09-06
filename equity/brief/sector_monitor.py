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
  (never benchmarks). `fetch_sector_data()` stores the full matrix too
  (as a plain `{ticker: {ticker: corr}}` dict, so it survives the JSON
  cache — see that function), but `format_sector_section()` still doesn't
  render the raw N×N table: `_format_correlation_clusters()` and
  `_format_top_hedges()` reduce it to concentration clusters (≥3 members,
  greedy-clustered on `CORRELATION_CONCENTRATION_THRESHOLD`) and the top 5
  natural hedges, pointing to `/discuss PORTFOLIO` for the full matrix.
- `compute_sector_concentration()` — count-of-positions breakdown by each
  position's `sector` field. No longer used by `format_sector_section()`,
  which now shows sector exposure weighted by `size_pct` instead (see
  `_format_sector_exposure()`) now that positions carry real IBKR-imported
  sizes; kept computing/returned in `fetch_sector_data()`'s result in case
  a future caller wants the count-based view.
- `fetch_etf_flows_stub()` — IBKR stub, same shape/pattern as
  `equity.portfolio.monitor.get_ibkr_positions()`.
- `fetch_options_activity()` — front-month options chain per position
  ticker: ATM-average implied vol as an "iv30" proxy (there's no separate
  30-day-constant-maturity fetch here), put/call volume ratio, and a
  `signal` (see `classify_options()`) describing whether that combination
  of IV and P/C is unusual enough to trust.

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
  that. Two whole-chain liquidity gates (`_clears_liquidity_gates()`)
  replace it instead of a single-strike check: `put_call_ratio` is only
  fed into `classify_options()` once the *entire* chain clears total
  volume (calls + puts) exceeding `MIN_TOTAL_VOLUME` contracts (filters
  out illiquid names like UMAC, where a handful of contracts can swing the
  ratio to an extreme), and total volume exceeding `VOLUME_OI_RATIO`x
  total open interest (filters out a chain that's merely being
  priced/quoted rather than actively traded). A name that fails either
  gate is classified on IV alone (`put_call_ratio` passed to
  `classify_options()` as `None`) regardless of how extreme its raw ratio
  looks — e.g. CEG's 1.6 P/C ratio contributes nothing to its `signal`
  because the chain's volume doesn't clear the gates.

  `classify_options()` itself (see `market_config.OPTIONS_*` for the
  thresholds) is a straight IV30/P-C lookup with no state of its own —
  the liquidity gating above is entirely the caller's (`fetch_options_activity()`'s)
  responsibility, so the function stays a pure, independently testable
  classifier.

`newly_crossed_pairs` (pairs whose correlation crossed 0.70 since the
*last* run, tracked via `equity/data/cache/concentration_history.json` —
there's no scheduled weekly cadence enforced anywhere in this codebase
yet, so this is really "since the last time this ran," not a strict
calendar week) is still computed and returned by `fetch_sector_data()`
but not currently surfaced by `format_sector_section()`; the cluster view
above supersedes it as the brief's concentration signal.
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
    OPTIONS_IV_ELEVATED,
    OPTIONS_IV_EXTREME,
    OPTIONS_PC_CALL_HEAVY,
    OPTIONS_PC_EXTREME_PUT,
    OPTIONS_PC_PUT_HEAVY,
    POSITION_SECTOR_MAP,
    SECTOR_WEIGHT_CONCENTRATION_PCT,
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

# Liquidity gates feeding classify_options() (see module docstring /
# _clears_liquidity_gates()): a chain's put/call ratio is only trusted
# once the whole chain — not a single strike — clears both a minimum
# absolute volume and a minimum volume relative to open interest. The
# actual IV/P-C classification thresholds live in market_config
# (OPTIONS_IV_ELEVATED etc.) since they're shared, decision-relevant
# thresholds, not a display-only convention.
MIN_TOTAL_VOLUME = 5000
VOLUME_OI_RATIO = 1.5

# Correlation-cluster display threshold — local to this module (see
# _format_correlation_clusters()): a cluster whose average intra-cluster
# correlation exceeds this is labeled HIGH risk rather than MODERATE.
# Distinct from CORRELATION_CONCENTRATION_THRESHOLD (0.70), which decides
# cluster *membership* itself.
CLUSTER_HIGH_RISK_CORR = 0.80


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


def _clears_liquidity_gates(total_volume: float, total_oi: float) -> bool:
    """True once a chain's whole-chain volume clears both gates (see module docstring):
    total volume > MIN_TOTAL_VOLUME contracts, and total volume >
    VOLUME_OI_RATIOx total open interest. A chain that fails either gate
    is too thin to trust its put/call ratio — a handful of contracts can
    swing it to an extreme that means nothing.
    """
    return total_volume > MIN_TOTAL_VOLUME and total_volume > VOLUME_OI_RATIO * total_oi


def classify_options(iv30: float | None, pc_ratio: float | None) -> str:
    """Classifies options positioning based on IV30 and put/call volume ratio.

    Returns one of: NO_DATA, NORMAL, IV_ELEVATED, IV_EXTREME, PUT_HEAVY,
    EXTREME_PUT, CALL_HEAVY, or a ' + '-joined combination of an IV signal
    and a P/C signal (e.g. 'IV_ELEVATED + PUT_HEAVY').

    Pure lookup against market_config.OPTIONS_* thresholds — no liquidity
    gating of its own. Callers computing `pc_ratio` from a real options
    chain (fetch_options_activity()) must pass None instead of a raw ratio
    once `_clears_liquidity_gates()` says the chain is too thin to trust.
    """
    if iv30 is None and pc_ratio is None:
        return "NO_DATA"

    signals = []

    if iv30 is not None:
        if iv30 > OPTIONS_IV_EXTREME:
            signals.append("IV_EXTREME")
        elif iv30 > OPTIONS_IV_ELEVATED:
            signals.append("IV_ELEVATED")

    if pc_ratio is not None:
        if pc_ratio > OPTIONS_PC_EXTREME_PUT:
            signals.append("EXTREME_PUT")
        elif pc_ratio > OPTIONS_PC_PUT_HEAVY:
            signals.append("PUT_HEAVY")
        elif pc_ratio < OPTIONS_PC_CALL_HEAVY:
            signals.append("CALL_HEAVY")

    return " + ".join(signals) if signals else "NORMAL"


def fetch_options_activity(tickers: list[str]) -> dict:
    """Front-month options snapshot per ticker: {iv30, put_call_ratio, signal}.

    Never raises: a ticker with no options chain, or any fetch failure,
    still gets an entry (iv30/put_call_ratio None, signal 'NO_DATA')
    rather than being dropped from the result.
    """
    result: dict[str, dict] = {}

    for ticker in tickers:
        entry = {"iv30": None, "put_call_ratio": None, "signal": "NO_DATA"}

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
        gated_pc_ratio = entry["put_call_ratio"] if _clears_liquidity_gates(total_volume, total_oi) else None
        entry["signal"] = classify_options(entry["iv30"], gated_pc_ratio)

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
        corr_matrix = None
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
        # Nested {ticker: {ticker: corr}} rather than the DataFrame itself —
        # this dict round-trips through the JSON cache (_write_cache() below
        # / _load_cache()), unlike a DataFrame. format_sector_section()
        # rebuilds a DataFrame from it with pd.DataFrame(...) before use.
        "correlation_matrix": corr_matrix.to_dict() if corr_matrix is not None else None,
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

def _format_sector_exposure(sector_data: dict) -> str:
    """Aggregates POSITIONS by sector and shows portfolio weight % (not position count).

    Uses each position's `size_pct` — already IBKR-import-merged onto
    `positions_config.POSITIONS` by `positions.py`'s override mechanism,
    so there's no need to re-read positions_override.json here. `sector_data`
    isn't actually needed for this (weights come straight from POSITIONS)
    but is accepted for a consistent signature with the other _format_*
    helpers and in case a future caller wants to pass pre-aggregated data.
    """
    sector_weights: dict[str, float] = {}
    sector_tickers: dict[str, list[str]] = {}

    for ticker, pos in positions_config.POSITIONS.items():
        sector = pos.get("sector") or "Unknown"
        weight = pos.get("size_pct") or 0
        sector_weights[sector] = sector_weights.get(sector, 0) + weight
        sector_tickers.setdefault(sector, []).append(ticker)

    if not sector_weights:
        return "Sector Weights\n  No positions with sector/size data available."

    sorted_sectors = sorted(sector_weights.items(), key=lambda kv: -kv[1])

    lines = ["Sector Weights"]
    for sector, weight in sorted_sectors:
        tickers = sector_tickers.get(sector, [])
        flag = " ⚠️ concentrated" if weight > SECTOR_WEIGHT_CONCENTRATION_PCT else ""
        if len(tickers) <= 3:
            ticker_str = f"  ({', '.join(tickers)})"
        else:
            ticker_str = f"  ({len(tickers)} positions)"
        lines.append(f"  {sector:<28} {weight:>5.1f}%{flag}{ticker_str}")

    return "\n".join(lines)


def _format_correlation_clusters(corr_matrix: pd.DataFrame, threshold: float = CORRELATION_CONCENTRATION_THRESHOLD) -> str:
    """Identifies concentration clusters via simple greedy clustering.

    A cluster is a group where every pair has |correlation| >= threshold.
    Shows members and avg correlation. Only clusters with >= 3 members are
    surfaced — the point is "these move as a block," which two names
    already say via `_format_top_hedges()`'s inverse or don't need calling
    out at all.
    """
    tickers = list(corr_matrix.columns)
    visited = set()
    clusters = []

    for t in tickers:
        if t in visited:
            continue
        cluster = [t]
        for other in tickers:
            if other == t or other in visited:
                continue
            if all(abs(corr_matrix.loc[m, other]) >= threshold for m in cluster):
                cluster.append(other)
        if len(cluster) >= 3:
            pairs = [(a, b) for i, a in enumerate(cluster) for b in cluster[i + 1:]]
            avg_corr = sum(abs(corr_matrix.loc[a, b]) for a, b in pairs) / len(pairs) if pairs else 0
            clusters.append({"members": cluster, "avg_corr": avg_corr})
            visited.update(cluster)

    if not clusters:
        return "No significant concentration clusters detected."

    lines = [f"Concentration Clusters (≥3 positions, avg corr >{threshold:.2f})"]
    for c in sorted(clusters, key=lambda x: -x["avg_corr"]):
        members = ", ".join(c["members"])
        risk_label = "HIGH" if c["avg_corr"] > CLUSTER_HIGH_RISK_CORR else "MODERATE"
        lines.append(f"  ⚠️ [{risk_label}] {members}")
        lines.append(f"     Avg correlation: {c['avg_corr']:.2f} — move as a block on macro shock")

    return "\n".join(lines)


def _format_top_hedges(corr_matrix: pd.DataFrame, top_n: int = 5) -> str:
    """The `top_n` most negatively correlated position pairs — natural hedges."""
    tickers = list(corr_matrix.columns)
    hedge_pairs = []
    for i, a in enumerate(tickers):
        for b in tickers[i + 1:]:
            c = corr_matrix.loc[a, b]
            if pd.notna(c) and c < CORRELATION_HEDGE_THRESHOLD:
                hedge_pairs.append((a, b, float(c)))
    hedge_pairs.sort(key=lambda x: x[2])  # most negative first
    top = hedge_pairs[:top_n]
    if not top:
        return "No significant natural hedges detected."
    lines = [f"Top {len(top)} Natural Hedges"]
    for a, b, c in top:
        lines.append(f"  ✓ {a} ↔ {b}: {c:.2f}")
    return "\n".join(lines)


def _format_options_flags(options_data: dict) -> str:
    """Only the positions whose options `signal` is neither NORMAL nor NO_DATA."""
    flagged = {
        ticker: data for ticker, data in options_data.items()
        if data.get("signal", "NORMAL") not in ("NORMAL", "NO_DATA")
    }
    if not flagged:
        return "Options Activity: No unusual positioning detected across portfolio."

    lines = ["Options Flags (unusual positioning only)"]
    for ticker, data in sorted(flagged.items()):
        iv = data.get("iv30")
        pc = data.get("put_call_ratio")
        signal = data.get("signal", "")
        iv_str = f"IV30: {iv:.0f}%" if iv is not None else "IV: n/a"
        pc_str = f"P/C: {pc:.1f}" if pc is not None else "P/C: n/a"
        lines.append(f"  ⚠️ {ticker}: {iv_str} | {pc_str} | {signal}")
    return "\n".join(lines)


def format_sector_section(sector_data: dict) -> str:
    """Render `fetch_sector_data()`'s output dict as a Telegram-ready string. Never raises."""
    lines = ["🏭 SECTOR & CONCENTRATION", _DIVIDER]

    # Block 1: sector weights (aggregated, not per-position)
    lines.append(_format_sector_exposure(sector_data))
    lines.append("")

    # Blocks 2/3: clusters + top hedges, replacing the old N×N pair listing.
    # correlation_matrix is stored as a plain {ticker: {ticker: corr}} dict
    # (JSON-cache-safe — see fetch_sector_data()) and rebuilt into a
    # DataFrame here for the .loc-based cluster/hedge logic.
    corr_dict = sector_data.get("correlation_matrix")
    if corr_dict:
        corr_matrix = pd.DataFrame(corr_dict)
        if not corr_matrix.empty:
            lines.append(_format_correlation_clusters(corr_matrix))
            lines.append("")
            lines.append(_format_top_hedges(corr_matrix))
            lines.append("")
            lines.append("Full correlation matrix: /discuss PORTFOLIO")
            lines.append("")

    # Block 4: options flags only (non-NORMAL/NO_DATA positions)
    options_activity = sector_data.get("options_activity", {})
    lines.append(_format_options_flags(options_activity))

    warnings = sector_data.get("data_warnings") or []
    if warnings:
        lines.append("")
        lines.extend(f"⚠️ {w}" for w in warnings)

    lines.append("")
    lines.append(_DIVIDER)
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    sector_result = fetch_sector_data()
    print(format_sector_section(sector_result))
