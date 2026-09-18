"""Market-wide snapshot (equities, VIX, rates, FX, commodities, regime) for the morning brief.

One batched `yf.download()` call covers every ticker in
`market_config.EQUITY_BENCHMARKS`, `COMMODITY_TICKERS`, `RATE_TICKERS`, plus
`^VIX` — mirrors `equity.screener.price_filter`'s batching approach, just
for a handful of macro tickers instead of the Russell 1000. A second,
separate `period='5y'` batch covers the full Treasury curve
(`TREASURY_TICKERS`), FX pairs (`FX_TICKERS`), and extended commodities
(`COMMODITY_TICKERS_EXTENDED`) — these need 5 years of daily history for
the `HIGHLIGHT_MA_PERIODS`/`HIGHLIGHT_EXTREMES_PCT` moving-average and
52W/5Y-extreme flags (`compute_ma_flags()`), which the `'ytd'` batch above
doesn't carry enough history for early in the calendar year.

Every threshold used for regime classification (`REGIME_RULES`,
`VIX_ELEVATED`/`VIX_HIGH`/`VIX_EXTREME`) or MA/extremes highlighting
(`HIGHLIGHT_MA_PERIODS`, `HIGHLIGHT_MA_PROXIMITY_PCT`,
`HIGHLIGHT_EXTREMES_PCT`) is imported from `equity.config.market_config` —
nothing regime- or threshold-related is hardcoded here.

Fed funds effective rate + target range, the 2Y/20Y Treasury points, JGB
10Y, and each FX pair's foreign policy/market rate (for the CIP-derived
forwards) come from FRED via `fredapi`, cached to
`equity/data/cache/` for 24h — same cache-file shape (`fetched_at` +
payload, age-checked on read) as `eco_calendar.py`'s FOMC cache.

FX forwards (`fetch_fx_forwards()`) are CIP-derived, not dealer quotes —
`forward = spot * (1 + r_usd*days/360) / (1 + r_foreign*days/360)`. The
Treasury curve only carries 3M/2Y/5Y/10Y/20Y/30Y tenors (no 6M or 1Y
point), so the "nearest available" USD leg for a forward tenor is: 1M/3M/6M
-> the 3M bill, 1Y -> the 2Y note (the closest point we actually have on
either side). A tenor is skipped (not crashed) if its FRED foreign-rate
series is missing/unavailable — several of the configured FRED series IDs
are long-dormant/legacy codes that may 404 depending on the account's FRED
catalog access; see `_fetch_fred_series_cached()`.
"""

import json
import logging
import operator
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

import anthropic
import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

from equity.config.market_config import (
    COMMODITY_TICKERS,
    COMMODITY_TICKERS_EXTENDED,
    CREDIT_TICKERS,
    CROSS_ASSET_RATIOS,
    CRYPTO_TICKERS,
    EQUITY_BENCHMARKS,
    EQUITY_FUTURES,
    FX_FORWARD_FOREIGN_RATES,
    FX_FORWARD_TENORS,
    FX_TENOR_DAYS,
    FX_TICKERS,
    HIGHLIGHT_EXTREMES_PCT,
    HIGHLIGHT_MA_PERIODS,
    HIGHLIGHT_MA_PROXIMITY_PCT,
    INTERNATIONAL_INDICES,
    JGB_10Y_FRED_SERIES,
    RATE_TICKERS,
    REGIME_RULES,
    SIGNAL_THRESHOLDS,
    TREASURY_FRED_SERIES,
    TREASURY_SPREADS,
    TREASURY_TICKERS,
    VIX_ELEVATED,
    VIX_EXTREME,
    VIX_HIGH,
    VOLATILITY_TICKERS,
)
from equity.brief.eco_calendar import fetch_fomc_dates
from equity.config.positions import POSITIONS
from equity.data.price_cache import price_cache
from equity.data.yfinance_utils import yf_download

load_dotenv()

logger = logging.getLogger(__name__)

FRED_API_KEY = os.getenv("FRED_API_KEY", "")

# Same model string as brief_synthesizer.MODEL / telegram.advisor.MODEL —
# defined locally rather than imported from either so this module's only
# hard dependency for its own data-fetch functions (fetch_market_snapshot,
# fetch_global_signals, ...) stays fredapi/yfinance/pandas; a missing
# ANTHROPIC_API_KEY should never block those from importing or running.
_MACRO_INTEL_MODEL = "claude-sonnet-4-6"

_DIVIDER = "━━━━━━━━━━━━━━━━━━━━━━━━"

_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache"
_FED_FUNDS_CACHE_PATH = _CACHE_DIR / "fed_funds.json"
FED_FUNDS_CACHE_HOURS = 24
FRED_SERIES_CACHE_HOURS = 24

_MACRO_INTEL_CACHE_PATH = _CACHE_DIR / "macro_intel.json"
MACRO_INTEL_CACHE_HOURS = 6

# Hourly cap on *cold* (cache-miss) macro-intel fetches — each cold fetch is
# 5 searches x 2 Claude calls (search + grounded-extraction) = 10 calls.
# The 6h cache already bounds this to ~4 cold fetches/day under normal use,
# but both build_morning_brief() and send_prices('all') can hit a cold
# cache independently (a fresh brief, then /prices before it's cached) —
# same failure shape advisor.py's WEB_FUNDAMENTALS_MAX_PER_HOUR guards
# against for _fetch_web_fundamentals(). This is the same kind of guard,
# scaled for one shared module-level cache instead of one per ticker.
MACRO_INTEL_MAX_PER_HOUR = 4
_macro_intel_call_times: list[float] = []

# Market-move thresholds that force a fresh macro-intel fetch even when the
# 6h cache (MACRO_INTEL_CACHE_HOURS) is still within its window — see
# _should_invalidate_cache(). A big move usually means a new narrative has
# emerged since the cache was written, and waiting out the rest of the
# window means the brief keeps citing stale color through the move.
MACRO_INTEL_VIX_SPIKE_PCT = 5.0        # VIX up >5% vs prior close: fear spike
MACRO_INTEL_FUTURES_DROP_PCT = -1.5    # ES=F/NQ=F down >1.5%: risk-off event
MACRO_INTEL_BRENT_SPIKE_PCT = 3.0      # BZ=F up >3%: energy event
MACRO_INTEL_POSITION_DROP_PCT = -5.0   # any held position down >5%: idiosyncratic event

# How many days ahead of an FOMC meeting the dedicated fomc_preview search
# (see fetch_macro_intelligence()) turns on. Deliberately wider than
# eco_calendar.FOMC_PROXIMITY_DAYS (2) — that constant gates position-size
# reduction close to the meeting; this just decides when a preview search
# is worth the extra API calls, so a full "Fed week" window is more useful.
MACRO_INTEL_FOMC_PREVIEW_WINDOW_DAYS = 5

VIX_TICKER = "^VIX"
TREASURY_5Y_PERIOD = "5y"
FRED_LOOKBACK_DAYS = 5 * 365 + 30  # ~5y + slack, for MA200/5Y-extreme series

# Display labels — presentation only, not thresholds, so not sourced from
# market_config (which has none for these). The legacy COMMODITY_TICKERS
# dict itself (DXY/Gold/Oil/Copper) is still fetched — DXY feeds
# dxy_change_1d for regime classification — but no longer has a display
# section of its own now that Commodities/FX render from the extended
# set (see "Legacy commodities/DXY" block in fetch_market_snapshot()).
_EXTENDED_COMMODITY_FORMATS = {
    "BZ=F": {"label": "Brent",    "prefix": "$", "decimals": 1},
    "CL=F": {"label": "WTI",      "prefix": "$", "decimals": 1},
    "NG=F": {"label": "Nat Gas",  "prefix": "$", "decimals": 2},
    "GC=F": {"label": "Gold",     "prefix": "$", "decimals": 0},
    "SI=F": {"label": "Silver",   "prefix": "$", "decimals": 1},
    "PL=F": {"label": "Platinum", "prefix": "$", "decimals": 0},
    "HG=F": {"label": "Copper",   "prefix": "$", "decimals": 2},
    "URA":  {"label": "URA",      "prefix": "$", "decimals": 1},
}

# Nearest-available USD leg on the Treasury curve for each FX-forward
# tenor — see module docstring (no 6M/1Y point on TREASURY_TICKERS/
# TREASURY_FRED_SERIES).
_FORWARD_TENOR_USD_LEG = {"1M": "3M", "3M": "3M", "6M": "3M", "1Y": "2Y"}

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
# Highlighting — moving-average proximity / 5Y extremes (shared across
# rates, spreads, FX, commodities, positions — see performance_tracker.py
# for the positions-side usage)
# ---------------------------------------------------------------------------

def compute_ma_flags(series: pd.Series, current_value: float) -> list[str]:
    """Flag strings for proximity to SMA(20/50/200) and 5Y high/low.

    Uses `HIGHLIGHT_MA_PROXIMITY_PCT`/`HIGHLIGHT_EXTREMES_PCT` from
    `market_config`. Never raises: a short series just skips whichever
    checks it doesn't have enough history for.
    """
    flags = []
    for period in HIGHLIGHT_MA_PERIODS:
        if len(series) >= period:
            ma = series.rolling(period).mean().iloc[-1]
            if ma and pd.notna(ma) and abs(current_value - ma) / abs(ma) * 100 <= HIGHLIGHT_MA_PROXIMITY_PCT:
                flags.append(f"near {period}D MA")
    if len(series) > 20:
        high_5y = series.max()
        low_5y = series.min()
        if high_5y and abs(current_value - high_5y) / abs(high_5y) * 100 <= HIGHLIGHT_EXTREMES_PCT:
            flags.append("near 5Y high")
        if low_5y and abs(current_value - low_5y) / abs(low_5y) * 100 <= HIGHLIGHT_EXTREMES_PCT:
            flags.append("near 5Y low")
    return flags


def format_flag(flags: list[str]) -> str:
    return f" ⚠️ {', '.join(flags)}" if flags else ""


# ---------------------------------------------------------------------------
# yfinance batch fetch
# ---------------------------------------------------------------------------

def _download_batch(tickers: list[str], period: str = "ytd") -> pd.DataFrame | None:
    try:
        data = yf_download(
            tickers,
            period=period,
            auto_adjust=True,
            group_by="ticker",
            threads=True,
            progress=False,
        )
    except Exception as exc:  # yfinance can raise a variety of things on network/format issues
        logger.warning("yf.download failed for market snapshot batch (period=%s): %s", period, exc)
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
# FRED — generic 24h series cache (fed funds target/range, plus every
# other FRED series this module pulls: DGS2/DGS20, JGB 10Y, FX forward
# foreign rates)
# ---------------------------------------------------------------------------

def _fred_series_cache_path(series_id: str) -> Path:
    safe = series_id.replace("/", "_")
    return _CACHE_DIR / f"fred_{safe}.json"


def _load_fred_series_cache(series_id: str) -> dict | None:
    path = _fred_series_cache_path(series_id)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            cache = json.load(f)
        fetched_at = datetime.fromisoformat(cache["fetched_at"])
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.warning("FRED cache for %s unreadable, ignoring: %s", series_id, exc)
        return None
    if (datetime.now() - fetched_at).total_seconds() > FRED_SERIES_CACHE_HOURS * 3600:
        return None
    return cache


def _write_fred_series_cache(series_id: str, series: pd.Series) -> None:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "fetched_at": datetime.now().isoformat(),
            "index": [d.isoformat() for d in series.index],
            "values": [float(v) for v in series.values],
        }
        with open(_fred_series_cache_path(series.name or "series"), "w") as f:
            json.dump(payload, f)
    except OSError as exc:
        logger.warning("Failed to write FRED cache for %s: %s", series.name, exc)


def _fetch_fred_series_cached(
    series_id: str, fred: "Fred | None", data_warnings: list[str], start: date | None = None
) -> pd.Series | None:
    """A FRED series (full history back to `start`, default ~5Y), 24h disk-cached.

    Never raises: a missing FRED client, an unknown/discontinued series
    id, or any request failure just appends to `data_warnings` and
    returns None.
    """
    cached = _load_fred_series_cache(series_id)
    if cached is not None:
        try:
            idx = pd.to_datetime(cached["index"])
            return pd.Series(cached["values"], index=idx, name=series_id)
        except (KeyError, ValueError) as exc:
            logger.warning("FRED cache for %s malformed, refetching: %s", series_id, exc)

    if fred is None:
        data_warnings.append(f"FRED_API_KEY not set — {series_id} unavailable")
        return None

    start = start or (date.today() - timedelta(days=FRED_LOOKBACK_DAYS))
    try:
        raw = fred.get_series(series_id, observation_start=start.isoformat())
    except Exception as exc:
        logger.warning("FRED series fetch failed for %s: %s", series_id, exc)
        data_warnings.append(f"{series_id}: FRED fetch failed: {exc}")
        return None

    series = raw.dropna()
    if series.empty:
        data_warnings.append(f"{series_id}: FRED returned no data")
        return None

    series.name = series_id
    _write_fred_series_cache(series_id, series)
    return series


def _fetch_fed_funds(fred: "Fred | None", data_warnings: list[str]) -> dict | None:
    """(effective_rate, target_upper, target_lower) from FRED, cache-first. None (+ warning) on failure."""
    cached = _load_fed_funds_cache()
    if cached is not None:
        return cached

    if fred is None:
        data_warnings.append("FRED_API_KEY not set — Fed funds rate unavailable")
        return None

    try:
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


# ---------------------------------------------------------------------------
# Treasury curve + spreads
# ---------------------------------------------------------------------------

def _fetch_treasury_curve(five_yr_data: pd.DataFrame | None, fred: "Fred | None", data_warnings: list[str]) -> dict:
    """Every tenor in `TREASURY_TICKERS` (yfinance) + `TREASURY_FRED_SERIES` (FRED) + JGB 10Y.

    Each entry: {'yield_pct', 'change_1d_bps', 'ma_flags'}. A tenor whose
    fetch fails is simply absent from the result (+ a data_warnings entry)
    rather than raising.
    """
    curve: dict[str, dict] = {}

    for ticker, tenor in TREASURY_TICKERS.items():
        close = _ticker_close(five_yr_data, ticker)
        if close is None or len(close) < 2:
            data_warnings.append(f"{ticker} ({tenor}): price data unavailable")
            continue
        yield_pct = float(close.iloc[-1])
        change_1d_bps = (yield_pct - float(close.iloc[-2])) * 100
        curve[tenor] = {
            "yield_pct": yield_pct,
            "change_1d_bps": change_1d_bps,
            "ma_flags": compute_ma_flags(close, yield_pct),
        }

    for series_id, tenor in TREASURY_FRED_SERIES.items():
        series = _fetch_fred_series_cached(series_id, fred, data_warnings)
        if series is None or len(series) < 2:
            continue
        yield_pct = float(series.iloc[-1])
        change_1d_bps = (yield_pct - float(series.iloc[-2])) * 100
        curve[tenor] = {
            "yield_pct": yield_pct,
            "change_1d_bps": change_1d_bps,
            "ma_flags": compute_ma_flags(series, yield_pct),
        }

    jgb_monthly = _fetch_fred_series_cached(JGB_10Y_FRED_SERIES, fred, data_warnings)
    if jgb_monthly is not None and not jgb_monthly.empty:
        # Forward-fill monthly -> daily so it lines up on a comparable axis
        # to the other (daily) tenors for MA/extremes purposes.
        daily_index = pd.date_range(jgb_monthly.index.min(), date.today(), freq="D")
        jgb_daily = jgb_monthly.reindex(daily_index).ffill().dropna()
        if len(jgb_daily) >= 2:
            yield_pct = float(jgb_daily.iloc[-1])
            change_1d_bps = (yield_pct - float(jgb_daily.iloc[-2])) * 100
            curve["JGB10Y"] = {
                "yield_pct": yield_pct,
                "change_1d_bps": change_1d_bps,
                "ma_flags": compute_ma_flags(jgb_daily, yield_pct),
                "monthly": True,
            }

    return curve


def _fetch_treasury_spreads(curve: dict) -> dict:
    """(long - short) * 100bps for every pair in `TREASURY_SPREADS`, from the already-fetched curve.

    Spread-level 1D change / MA flags aren't computed here (that would
    need aligned daily long/short spread history, which the point-in-time
    `curve` dict doesn't carry) — only the current spread level plus
    each leg's own 1D change, which is enough to explain a spread move.
    """
    spreads: dict[str, dict] = {}
    for name, (short_tenor, long_tenor) in TREASURY_SPREADS.items():
        short = curve.get(short_tenor)
        long = curve.get(long_tenor)
        if short is None or long is None:
            continue
        spread_bps = (long["yield_pct"] - short["yield_pct"]) * 100
        change_1d_bps = long["change_1d_bps"] - short["change_1d_bps"]
        spreads[name] = {"spread_bps": spread_bps, "change_1d_bps": change_1d_bps}
    return spreads


# ---------------------------------------------------------------------------
# FX spot + forwards (CIP-derived)
# ---------------------------------------------------------------------------

def fetch_fx_data(five_yr_data: pd.DataFrame | None = None, data_warnings: list[str] | None = None) -> dict:
    """Every pair in `market_config.FX_TICKERS`: {'label', 'price', 'change_1d_pct', 'ma_flags'}.

    Reuses `five_yr_data` (the combined Treasury/FX/commodities 5Y batch)
    if supplied; otherwise issues its own batched `yf.download()`.
    """
    data_warnings = data_warnings if data_warnings is not None else []
    if five_yr_data is None:
        five_yr_data = _download_batch(list(FX_TICKERS), period=TREASURY_5Y_PERIOD)

    result: dict[str, dict] = {}
    for ticker, label in FX_TICKERS.items():
        close = _ticker_close(five_yr_data, ticker)
        if close is None or len(close) < 2:
            data_warnings.append(f"{ticker} ({label}): price data unavailable")
            continue
        price = float(close.iloc[-1])
        change_1d_pct = (price / float(close.iloc[-2]) - 1) * 100
        result[ticker] = {
            "label": label,
            "price": price,
            "change_1d_pct": change_1d_pct,
            "ma_flags": compute_ma_flags(close, price),
        }
    return result


def fetch_fx_forwards(fx_data: dict, treasury_curve: dict, fred: "Fred | None", data_warnings: list[str]) -> dict:
    """CIP-derived forward points/premium for each pair in `FX_FORWARD_FOREIGN_RATES`.

    forward = spot * (1 + r_usd*days/360) / (1 + r_foreign*days/360).
    A tenor is skipped (not the whole pair) if its FRED foreign-rate series
    or the nearest USD Treasury leg is unavailable — never raises. See
    module docstring for the tenor->USD-leg mapping and the caveat about
    some configured FRED series ids being legacy/discontinued codes.
    """
    result: dict[str, dict] = {}

    for pair, foreign_rates in FX_FORWARD_FOREIGN_RATES.items():
        spot_entry = fx_data.get(pair)
        if spot_entry is None:
            continue
        spot = spot_entry["price"]
        # USD/JPY, USD/CAD, USD/CHF quote USD as the base currency — CIP
        # still applies the same way (foreign currency is the JPY/CAD/CHF
        # leg either way), so no direction-flip needed here.

        pair_forwards: dict[str, dict] = {}
        for tenor in FX_FORWARD_TENORS:
            foreign_series_id = foreign_rates.get(tenor)
            usd_leg = _FORWARD_TENOR_USD_LEG.get(tenor)
            if foreign_series_id is None or usd_leg is None:
                continue
            usd_entry = treasury_curve.get(usd_leg)
            if usd_entry is None:
                continue

            foreign_series = _fetch_fred_series_cached(foreign_series_id, fred, data_warnings)
            if foreign_series is None or foreign_series.empty:
                continue

            r_usd = usd_entry["yield_pct"] / 100
            r_foreign = float(foreign_series.iloc[-1]) / 100
            days = FX_TENOR_DAYS.get(tenor)
            if days is None:
                continue

            try:
                forward = spot * (1 + r_usd * days / 360) / (1 + r_foreign * days / 360)
            except ZeroDivisionError:
                continue

            # JPY pairs quote to 2 decimals (1 pip = 0.01), everything else
            # to 4 (1 pip = 0.0001) — same convention forex desks use.
            pip_multiplier = 100 if "JPY" in pair else 10000
            pair_forwards[tenor] = {
                "forward": forward,
                "forward_points_pips": (forward - spot) * pip_multiplier,
                "premium_pct": (forward / spot - 1) * 100,
            }

        if pair_forwards:
            result[pair] = pair_forwards

    return result


# ---------------------------------------------------------------------------
# Extended commodities
# ---------------------------------------------------------------------------

def _fetch_extended_commodities(five_yr_data: pd.DataFrame | None, data_warnings: list[str]) -> dict:
    result: dict[str, dict] = {}
    for ticker, label in COMMODITY_TICKERS_EXTENDED.items():
        close = _ticker_close(five_yr_data, ticker)
        if close is None or len(close) < 2:
            data_warnings.append(f"{ticker} ({label}): price data unavailable")
            continue
        price = float(close.iloc[-1])
        change_1d_pct = (price / float(close.iloc[-2]) - 1) * 100
        result[ticker] = {
            "label": label,
            "price": price,
            "change_1d_pct": change_1d_pct,
            "ma_flags": compute_ma_flags(close, price),
        }

    brent = result.get("BZ=F")
    wti = result.get("CL=F")
    brent_wti_spread = (brent["price"] - wti["price"]) if brent and wti else None

    return {"commodities": result, "brent_wti_spread": brent_wti_spread}


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
    """Batch-fetch equities/VIX/rates/FX/commodities, classify the day's regime, and return one dict.

    Never raises: a failed batch download, a missing ticker, or a failed
    Fed funds/FRED fetch is recorded in `data_warnings` and the affected
    section is omitted/left partial rather than crashing the whole
    snapshot.
    """
    data_warnings: list[str] = []

    # 'ytd' batch — equities/VIX/DXY (regime inputs only need today vs.
    # yesterday, so 'ytd' history is plenty; keeping this batch separate
    # from the 5Y one below avoids re-fetching a year of daily bars for a
    # comparison that only ever looks at the last two rows).
    tickers = list(EQUITY_BENCHMARKS) + list(COMMODITY_TICKERS) + list(RATE_TICKERS) + [VIX_TICKER]
    data = _download_batch(tickers, period="ytd")
    if data is None:
        data_warnings.append("yf.download returned no data for the market snapshot batch")

    # '5y' batch — Treasury curve, FX pairs, extended commodities: all need
    # enough history for compute_ma_flags()'s SMA200/5Y-extreme checks.
    five_yr_tickers = list(TREASURY_TICKERS) + list(FX_TICKERS) + list(COMMODITY_TICKERS_EXTENDED)
    five_yr_data = _download_batch(five_yr_tickers, period=TREASURY_5Y_PERIOD)
    if five_yr_data is None:
        data_warnings.append("yf.download returned no data for the 5Y treasury/FX/commodities batch")

    fred = Fred(api_key=FRED_API_KEY) if FRED_API_KEY else None
    if fred is None:
        data_warnings.append("FRED_API_KEY not set — 2Y/20Y Treasury, JGB, and FX forwards unavailable")

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

    # --- Legacy rates (kept for regime inputs — see _CONDITION_GETTERS) ---
    yield_10y = yield_10y_change_bps = yield_3m = None
    for ticker in RATE_TICKERS:
        close = _ticker_close(data, ticker)
        if close is None or len(close) < 2:
            continue
        yield_pct = float(close.iloc[-1])
        change_1d_bps = (yield_pct - float(close.iloc[-2])) * 100
        if ticker == "^TNX":
            yield_10y, yield_10y_change_bps = yield_pct, change_1d_bps
        elif ticker == "^IRX":
            yield_3m = yield_pct

    yield_curve_2s10s_approx = None
    if yield_10y is not None and yield_3m is not None:
        yield_curve_2s10s_approx = yield_10y - yield_3m  # 3M used as 2Y proxy — see module docstring

    # --- Legacy commodities/DXY (kept for regime inputs + old display fields) ---
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

    # --- Treasury curve + spreads ---
    treasury_curve = _fetch_treasury_curve(five_yr_data, fred, data_warnings)
    treasury_spreads = _fetch_treasury_spreads(treasury_curve)

    # --- FX spot + forwards ---
    fx_data = fetch_fx_data(five_yr_data, data_warnings)
    fx_forwards = fetch_fx_forwards(fx_data, treasury_curve, fred, data_warnings)

    # --- Extended commodities ---
    extended = _fetch_extended_commodities(five_yr_data, data_warnings)

    # --- Fed funds ---
    fed_funds = _fetch_fed_funds(fred, data_warnings)

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
        "yield_curve_2s10s_approx": yield_curve_2s10s_approx,
        "commodities": commodities,
        "treasury_curve": treasury_curve,
        "treasury_spreads": treasury_spreads,
        "fx": fx_data,
        "fx_forwards": fx_forwards,
        "commodities_extended": extended["commodities"],
        "brent_wti_spread": extended["brent_wti_spread"],
        "fed_funds": fed_funds,
        "regime_flags": regime_flags,
        "as_of": datetime.now().isoformat(),
        "data_warnings": data_warnings,
    }


def _fmt_amount(value: float, prefix: str, decimals: int) -> str:
    return f"{prefix}{value:,.{decimals}f}"


def _format_treasury_line(tenor: str, entry: dict) -> str:
    monthly_note = "  (monthly)" if entry.get("monthly") else ""
    return (
        f"  {tenor:<5} {entry['yield_pct']:.2f}%  ({entry['change_1d_bps']:+.0f}bp)"
        f"{format_flag(entry.get('ma_flags', []))}{monthly_note}"
    )


def _format_spread_line(name: str, entry: dict) -> str:
    return f"  {name:<6} {entry['spread_bps']:+.0f}bp  ({entry['change_1d_bps']:+.0f}bp)"


def _format_fx_line(ticker: str, entry: dict, cross: bool = False) -> str:
    cross_note = "  (cross)" if cross else ""
    return (
        f"  {entry['label']:<7} {entry['price']:.4f}  {entry['change_1d_pct']:+.1f}%"
        f"{format_flag(entry.get('ma_flags', []))}{cross_note}"
    )


_FX_CROSS_PAIRS = {"EURJPY=X", "EURGBP=X"}


def _format_fx_forwards_line(pair: str, label: str, tenors: dict) -> str:
    parts = [f"{t} {tenors[t]['forward_points_pips']:+.0f}pip" for t in FX_FORWARD_TENORS if t in tenors]
    return f"  {label:<8} " + "  ".join(parts) if parts else f"  {label:<8} n/a"


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

    # --- Rates (US Treasury) ---
    lines.append("")
    lines.append("Rates (US Treasury)")
    treasury_curve = snapshot.get("treasury_curve", {})
    tenor_order = ["3M", "2Y", "5Y", "10Y", "20Y", "30Y", "JGB10Y"]
    for tenor in tenor_order:
        entry = treasury_curve.get(tenor)
        label = "JGB" if tenor == "JGB10Y" else tenor
        if entry is None:
            lines.append(f"  {label:<5} data unavailable")
            continue
        lines.append(_format_treasury_line(label, entry))

    fed_funds = snapshot.get("fed_funds")
    fed_str = f"{fed_funds['target_lower']:.2f}-{fed_funds['target_upper']:.2f}%" if fed_funds else "data unavailable"
    lines.append(f"  Fed:  {fed_str}")

    # --- Spreads ---
    lines.append("")
    lines.append("Spreads")
    treasury_spreads = snapshot.get("treasury_spreads", {})
    for name in TREASURY_SPREADS:
        entry = treasury_spreads.get(name)
        if entry is None:
            lines.append(f"  {name:<6} data unavailable")
            continue
        lines.append(_format_spread_line(name, entry))

    # --- FX ---
    lines.append("")
    lines.append("FX")
    fx_data = snapshot.get("fx", {})
    for ticker in FX_TICKERS:
        entry = fx_data.get(ticker)
        if entry is None:
            lines.append(f"  {FX_TICKERS[ticker]:<7} data unavailable")
            continue
        lines.append(_format_fx_line(ticker, entry, cross=ticker in _FX_CROSS_PAIRS))

    # --- FX forwards ---
    fx_forwards = snapshot.get("fx_forwards", {})
    if fx_forwards:
        lines.append("")
        lines.append("FX Forwards (CIP-derived — not dealer quotes)")
        for pair, tenors in fx_forwards.items():
            lines.append(_format_fx_forwards_line(pair, FX_TICKERS.get(pair, pair), tenors))

    # --- Commodities ---
    lines.append("")
    lines.append("Commodities")
    commodities_ext = snapshot.get("commodities_extended", {})
    brent = commodities_ext.get("BZ=F")
    wti = commodities_ext.get("CL=F")
    spread = snapshot.get("brent_wti_spread")
    if brent and wti:
        spread_str = f"  | spread {'+' if spread >= 0 else ''}${spread:.1f}" if spread is not None else ""
        lines.append(
            f"  Brent  {_fmt_amount(brent['price'], '$', 1)}  {brent['change_1d_pct']:+.1f}%"
            f"{format_flag(brent.get('ma_flags', []))} | "
            f"WTI  {_fmt_amount(wti['price'], '$', 1)}  {wti['change_1d_pct']:+.1f}%"
            f"{format_flag(wti.get('ma_flags', []))}{spread_str}"
        )
    for ticker, fmt in _EXTENDED_COMMODITY_FORMATS.items():
        if ticker in ("BZ=F", "CL=F"):
            continue  # rendered together above
        entry = commodities_ext.get(ticker)
        label = fmt["label"]
        if entry is None:
            lines.append(f"  {label:<8} data unavailable")
            continue
        lines.append(
            f"  {label:<8} {_fmt_amount(entry['price'], fmt['prefix'], fmt['decimals']):>10}  "
            f"{entry['change_1d_pct']:+.1f}%{format_flag(entry.get('ma_flags', []))}"
        )

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


# ---------------------------------------------------------------------------
# Global signals — equity futures, crypto, volatility, international
# indices, credit proxies, and cross-asset ratios. All read from the
# shared price_cache (see that module's docstring) rather than fetched
# here — this module's own 5Y-history batches exist for MA/extremes on
# the tickers above, which none of these need.
# ---------------------------------------------------------------------------

# Local trading hours per international index (weekday-only, holidays not
# accounted for — same simplification price_cache.py's staleness estimate
# makes). Distinct from price_cache's own is_stale/session_label: this
# answers "is that index's market open right now," from the wall clock;
# is_stale answers "how old is the data we actually have," from the last
# bar's timestamp. Both are shown together for international indices since
# they answer different questions.
_INTL_SESSION_HOURS = {
    "^N225":     ("Asia/Tokyo", 9, 0, 15, 30),
    "^KS11":     ("Asia/Seoul", 9, 0, 15, 30),
    "^TWII":     ("Asia/Taipei", 9, 0, 13, 30),
    "^HSI":      ("Asia/Hong_Kong", 9, 30, 16, 0),
    "^GDAXI":    ("Europe/Berlin", 9, 0, 17, 30),
    "^FTSE":     ("Europe/London", 8, 0, 16, 30),
    "^FCHI":     ("Europe/Paris", 9, 0, 17, 30),
    "^STOXX50E": ("Europe/Berlin", 9, 0, 17, 30),
    "^AXJO":     ("Australia/Sydney", 10, 0, 16, 0),
    "^BSESN":    ("Asia/Kolkata", 9, 15, 15, 30),
}


def _get_market_session_status(index_ticker: str) -> str:
    """' 🟢 live' / ' ⚪ opens in Nm' / ' 🔴 closed' for `index_ticker`'s home
    exchange, right now — weekday-only, holidays not accounted for.
    '' if `index_ticker` isn't in `_INTL_SESSION_HOURS`.
    """
    import pytz

    if index_ticker not in _INTL_SESSION_HOURS:
        return ""

    tz_name, open_h, open_m, close_h, close_m = _INTL_SESSION_HOURS[index_ticker]
    local_now = datetime.now(pytz.utc).astimezone(pytz.timezone(tz_name))

    if local_now.weekday() >= 5:
        return " 🔴 closed"

    open_time = local_now.replace(hour=open_h, minute=open_m, second=0, microsecond=0)
    close_time = local_now.replace(hour=close_h, minute=close_m, second=0, microsecond=0)

    if open_time <= local_now <= close_time:
        return " 🟢 live"
    if local_now < open_time:
        mins_to_open = int((open_time - local_now).total_seconds() / 60)
        return f" ⚪ opens in {mins_to_open}min"
    return " 🔴 closed"


def fetch_global_signals() -> dict:
    """Global market signals from `price_cache`. Never raises: a ticker
    price_cache doesn't have (e.g. ^MOVE, frequently unavailable on
    yfinance) is simply absent from its section rather than erroring.
    """
    def get(ticker: str) -> dict:
        return price_cache.get(ticker) or {}

    # Each section keyed by ticker (not label) — format_global_signals()
    # looks the label up from the same market_config dict at render time.
    # Keying by label here would mean either mutating price_cache's own
    # cache entry to stash the ticker on it (get() returns the live dict,
    # not a copy) or losing the ticker entirely — neither is worth it just
    # to save one dict lookup downstream.
    result: dict = {"futures": {}, "crypto": {}, "volatility": {}, "international": {}, "credit": {}, "ratios": {}}

    for ticker in EQUITY_FUTURES:
        d = get(ticker)
        if d.get("price") is not None:
            result["futures"][ticker] = d

    for ticker in CRYPTO_TICKERS:
        d = get(ticker)
        if d.get("price") is not None:
            result["crypto"][ticker] = d

    for ticker in VOLATILITY_TICKERS:
        d = get(ticker)
        if d.get("price") is not None:
            result["volatility"][ticker] = d

    for ticker in INTERNATIONAL_INDICES:
        d = get(ticker)
        if d.get("price") is not None:
            result["international"][ticker] = d

    for ticker in CREDIT_TICKERS:
        d = get(ticker)
        if d.get("price") is not None:
            result["credit"][ticker] = d

    for ratio_name, (t1, t2, description) in CROSS_ASSET_RATIOS.items():
        d1, d2 = get(t1), get(t2)
        if not d1.get("price") or not d2.get("price"):
            continue
        ratio = d1["price"] / d2["price"]
        ratio_chg = None
        if d1.get("prev_close") and d2.get("prev_close"):
            prev_ratio = d1["prev_close"] / d2["prev_close"]
            if prev_ratio:
                ratio_chg = (ratio / prev_ratio - 1) * 100
        result["ratios"][ratio_name] = {"ratio": ratio, "change_pct": ratio_chg, "description": description}

    return result


def format_global_signals(signals: dict) -> str:
    """Render `fetch_global_signals()`'s output as a Telegram-ready string. Never raises."""
    lines = ["🌍 GLOBAL SIGNALS", _DIVIDER]

    if signals.get("futures"):
        lines.append("Equity Futures")
        for ticker, d in signals["futures"].items():
            label = EQUITY_FUTURES.get(ticker, ticker)
            chg = d.get("change_1d_pct", 0)
            price = d.get("price", 0)
            emoji = "🟢" if chg > 0 else "🔴" if chg < 0 else "⚪"
            lines.append(f"  {emoji} {label}: {price:,.0f} ({chg:+.2f}%)")
        lines.append("")

    if signals.get("volatility"):
        lines.append("Volatility")
        for ticker, d in signals["volatility"].items():
            label = VOLATILITY_TICKERS.get(ticker, ticker)
            price = d.get("price", 0)
            chg = d.get("change_1d_pct", 0)
            flag = ""
            if "VVIX" in label:
                if price > SIGNAL_THRESHOLDS["vvix_extreme"]:
                    flag = " 🔴 EXTREME"
                elif price > SIGNAL_THRESHOLDS["vvix_elevated"]:
                    flag = " ⚠️ elevated"
            elif "VIX" in label:
                if price > SIGNAL_THRESHOLDS["vix_high"]:
                    flag = " 🔴 HIGH"
                elif price > SIGNAL_THRESHOLDS["vix_elevated"]:
                    flag = " ⚠️ elevated"
            lines.append(f"  {label}: {price:.1f} ({chg:+.1f}%){flag}")
        lines.append("")

    if signals.get("crypto"):
        lines.append("Crypto")
        for ticker, d in signals["crypto"].items():
            label = CRYPTO_TICKERS.get(ticker, ticker)
            price = d.get("price", 0)
            chg = d.get("change_1d_pct", 0)
            emoji = "🟢" if chg > 0 else "🔴" if chg < 0 else "⚪"
            flag = " ⚠️" if abs(chg) > SIGNAL_THRESHOLDS["btc_move_pct"] else ""
            lines.append(f"  {emoji} {label}: ${price:,.0f} ({chg:+.1f}%){flag}")
        lines.append("")

    if signals.get("international"):
        lines.append("International Indices")
        for ticker, d in signals["international"].items():
            label = INTERNATIONAL_INDICES.get(ticker, ticker)
            price = d.get("price", 0)
            chg = d.get("change_1d_pct", 0)
            emoji = "🟢" if chg > 0 else "🔴" if chg < 0 else "⚪"
            flag = " ⚠️" if abs(chg) > SIGNAL_THRESHOLDS["intl_index_pct"] else ""
            session = _get_market_session_status(ticker)
            lines.append(f"  {emoji} {label}: {price:,.0f} ({chg:+.2f}%){flag}{session}")
        lines.append("")

    if signals.get("credit"):
        lines.append("Credit Proxies")
        for ticker, d in signals["credit"].items():
            label = CREDIT_TICKERS.get(ticker, ticker)
            price = d.get("price", 0)
            chg = d.get("change_1d_pct", 0)
            emoji = "🟢" if chg > 0 else "🔴" if chg < 0 else "⚪"
            lines.append(f"  {emoji} {label}: ${price:.2f} ({chg:+.2f}%)")
        lines.append("")

    if signals.get("ratios"):
        lines.append("Cross-Asset Signals")
        for ratio_name, data in signals["ratios"].items():
            ratio = data["ratio"]
            chg = data.get("change_pct")
            desc = data["description"]
            chg_str = f" ({chg:+.2f}%)" if chg is not None else ""
            if ratio_name == "copper_gold":
                signal = "↑ growth" if (chg or 0) > 0 else "↓ growth concern"
            elif ratio_name == "silver_gold":
                signal = "↑ risk-on" if (chg or 0) > 0 else "↓ risk-off"
            elif ratio_name == "vix_vvix":
                signal = "⚠️ vol spike expected" if ratio > 0.25 else "normal"
            else:
                signal = ""
            arrow = f" → {signal}" if signal else ""
            lines.append(f"  {desc}: {ratio:.4f}{chg_str}{arrow}")
        lines.append("")

    lines.append(_DIVIDER)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Macro intelligence — the web-searched narrative layer for Section 1.
#
# Complements (doesn't duplicate) eco_calendar.py's FRED-sourced release
# *dates*: this adds the consensus/prior/"what a beat means" color FRED's
# release calendar doesn't carry, plus geopolitical/central-bank/overnight-
# session/analyst-action context that has no quantitative source at all in
# this module. Single-call search-and-synthesize shape (see
# _macro_search_and_synthesize()): Claude reads the actual web_search
# results and writes the briefing prose directly, in the same turn — no
# second grounded-extraction pass. Unlike
# equity.telegram.advisor._web_search_and_extract()'s two-call
# search-then-extract shape, this trades the extraction pass's "only state
# facts literally in source" guardrail for denser, more connected prose;
# `system` still tells the model never to fabricate.
# ---------------------------------------------------------------------------

_macro_intel_client: "anthropic.Anthropic | None" = None


def _get_macro_intel_client() -> "anthropic.Anthropic | None":
    """Lazily builds the Anthropic client — never at import time, so a
    missing ANTHROPIC_API_KEY can't break importing this module's (much
    more commonly used) plain data-fetch functions. Returns None if unset.
    """
    global _macro_intel_client
    if _macro_intel_client is not None:
        return _macro_intel_client
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    _macro_intel_client = anthropic.Anthropic(api_key=api_key)
    return _macro_intel_client


def _load_macro_intel_cache() -> dict | None:
    """None on any miss (absent/corrupt/stale file) — same fetched_at +
    payload shape and age-check as _load_fed_funds_cache().
    """
    if not _MACRO_INTEL_CACHE_PATH.exists():
        return None
    try:
        with open(_MACRO_INTEL_CACHE_PATH) as f:
            cache = json.load(f)
        fetched_at = datetime.fromisoformat(cache["fetched_at"])
        results = dict(cache["results"])
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.warning("macro_intel.json cache unreadable, ignoring: %s", exc)
        return None

    if (datetime.now() - fetched_at).total_seconds() > MACRO_INTEL_CACHE_HOURS * 3600:
        return None
    results["_fetched_at"] = cache["fetched_at"]
    return results


def _write_macro_intel_cache(results: dict, fetched_at_iso: str) -> None:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(_MACRO_INTEL_CACHE_PATH, "w") as f:
            json.dump({"fetched_at": fetched_at_iso, "results": results}, f, indent=2)
    except OSError as exc:
        logger.warning("Failed to write macro_intel.json cache: %s", exc)


def _macro_intel_budget_ok() -> bool:
    """Hourly cap on *cold* (cache-miss) fetches — see MACRO_INTEL_MAX_PER_HOUR."""
    now = time.time()
    _macro_intel_call_times[:] = [t for t in _macro_intel_call_times if now - t < 3600]
    if len(_macro_intel_call_times) >= MACRO_INTEL_MAX_PER_HOUR:
        return False
    _macro_intel_call_times.append(now)
    return True


def _get_held_tickers() -> str:
    """Comma-separated top-15 held tickers, for the synthesis system prompt."""
    try:
        return ", ".join(list(POSITIONS.keys())[:15])
    except Exception:
        return ""


def _macro_search_and_synthesize(
    query: str,
    context_prompt: str,
    max_tokens: int = 500,
    regime_flags: list[str] | None = None,
) -> str:
    """Single-call pattern: Claude searches and synthesizes in one turn.
    No extraction pass — Claude reads the actual web_search results and
    writes a direct, actionable briefing paragraph.

    This produces better output than the previous search-then-extract
    shape because:
    - Claude sees the actual web content, not a formatted summary
    - No "not found" from over-strict extraction rules
    - Claude can connect multiple sources into a coherent narrative
    - Output is prose ready for the brief, not structured fields

    `regime_flags` (from fetch_macro_intelligence()'s caller) is folded
    into the system prompt when present, so the synthesis interprets news
    through the current regime lens rather than in a vacuum.

    Never raises — returns '' on any failure (no client configured,
    search error, or an empty response) so one search failing doesn't
    take the others down with it.
    """
    client = _get_macro_intel_client()
    if client is None:
        return ""
    regime_str = ""
    if regime_flags:
        regime_str = (
            f"Current market regime flags: {', '.join(regime_flags)}. "
            f"Interpret news through this lens. "
        )
    try:
        response = client.messages.create(
            model=_MACRO_INTEL_MODEL,
            max_tokens=max_tokens,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            system=(
                f"You are writing the macro intelligence section of a morning "
                f"brief for a discretionary portfolio manager. "
                f"Today is {datetime.now().strftime('%A, %B %d, %Y')}. "
                f"The portfolio holds: {_get_held_tickers()}. "
                f"{regime_str}"
                f"Search for the information requested. "
                f"Write a direct, dense briefing — not a structured extraction. "
                f"Name specific price levels, specific companies, specific events. "
                f"Connect events to held positions by ticker. "
                f"If you find conflicting information, note it and use the more "
                f"credible source. "
                f"Never fabricate. If genuinely nothing found, say so in one line."
            ),
            messages=[{
                "role": "user",
                "content": f"{context_prompt}\n\nSearch query: {query}",
            }],
        )
        text = "\n".join(
            b.text for b in response.content if hasattr(b, "text") and b.text
        ).strip()
        return text if text else ""
    except Exception as exc:
        logger.warning("_macro_search_and_synthesize failed: %s", exc)
        return ""


def _is_fomc_preview_week() -> bool:
    """True when the next FOMC meeting is within
    MACRO_INTEL_FOMC_PREVIEW_WINDOW_DAYS — gates the dedicated fomc_preview
    search in fetch_macro_intelligence(). Reuses eco_calendar's own FOMC
    date source (scrape-with-fallback, cached — see fetch_fomc_dates())
    rather than re-deriving meeting dates here. Never raises: any failure
    (including an empty/malformed date list) just skips the extra search.
    """
    try:
        fomc_dates = fetch_fomc_dates()
        today = date.today()
        return any(
            0 <= (date.fromisoformat(d) - today).days <= MACRO_INTEL_FOMC_PREVIEW_WINDOW_DAYS
            for d in fomc_dates
        )
    except Exception as exc:
        logger.warning("_is_fomc_preview_week: failed, skipping fomc_preview search: %s", exc)
        return False


def _should_invalidate_cache() -> bool:
    """True if a large market move suggests a new narrative has emerged
    since the macro-intel cache was written — forces fetch_macro_intelligence()
    past a still-fresh (<MACRO_INTEL_CACHE_HOURS) cache to refetch.

    Reads price_cache rather than fetching anything itself, so this never
    adds its own network round-trip beyond price_cache's own (normal,
    already-happening) refresh cycle. Never raises: any failure (price_cache
    unavailable, a ticker missing from it) just means "don't invalidate" —
    a missed invalidation costs a stale narrative for up to the rest of the
    cache window, which is the pre-existing behavior; a false invalidation
    would burn part of the hourly cold-fetch budget for nothing.
    """
    try:
        vix = price_cache.get("^VIX")
        if vix and (vix.get("change_1d_pct") or 0) > MACRO_INTEL_VIX_SPIKE_PCT:
            logger.info("macro_intel cache invalidated: VIX spike")
            return True

        for ticker in ("ES=F", "NQ=F"):
            d = price_cache.get(ticker)
            if d and (d.get("change_1d_pct") or 0) < MACRO_INTEL_FUTURES_DROP_PCT:
                logger.info("macro_intel cache invalidated: %s down sharply", ticker)
                return True

        brent = price_cache.get("BZ=F")
        if brent and (brent.get("change_1d_pct") or 0) > MACRO_INTEL_BRENT_SPIKE_PCT:
            logger.info("macro_intel cache invalidated: Brent spike")
            return True

        for ticker in POSITIONS:
            d = price_cache.get(ticker)
            if d and (d.get("change_1d_pct") or 0) < MACRO_INTEL_POSITION_DROP_PCT:
                logger.info("macro_intel cache invalidated: %s down sharply", ticker)
                return True

        return False
    except Exception as exc:
        logger.warning("_should_invalidate_cache: failed, leaving cache as-is: %s", exc)
        return False


def fetch_macro_intelligence(regime_flags: list[str] | None = None) -> dict:
    """Seven to eight targeted web searches surfacing the day's lead
    market-moving story, central-bank, energy/geopolitical, overnight-
    session, tech/AI, analyst-action, and economic-release narrative
    context (plus a dedicated FOMC preview in the days before a meeting —
    see _is_fomc_preview_week()) — the narrative layer for Section 1
    (Global Markets), alongside
    fetch_market_snapshot()/fetch_global_signals()'s quantitative data.

    Cached MACRO_INTEL_CACHE_HOURS (6) hours in one shared file, keyed by
    fetched_at rather than a wall-clock bucket, so build_morning_brief()
    and send_prices('all') hitting this within the same window share one
    fetch instead of each maintaining their own (buggier) notion of "this
    hour". A still-fresh cache is discarded early anyway when
    _should_invalidate_cache() sees a large enough market move that a new
    narrative likely emerged since the cache was written — otherwise a
    move early in the window would sit un-narrated for the rest of it.
    The searches run concurrently, not sequentially — each is one
    search-and-synthesize call (see _macro_search_and_synthesize()), but
    a blocking Claude call with a live web search still takes several
    seconds, so running 7-8 of them back to back would push a cold fetch
    well past what a morning brief should wait on.

    Returns {} — never raises — if ANTHROPIC_API_KEY is unset, the hourly
    cold-fetch budget (MACRO_INTEL_MAX_PER_HOUR) is exhausted, or every
    search fails. Callers treat an empty/falsy result as "no macro intel
    layer this time" via format_macro_intelligence()'s content check.
    An empty/skipped result is never cached, so the next call (this
    hour's budget or key permitting) tries again rather than waiting out
    the full cache window for what wasn't actually fetched.
    """
    cached = _load_macro_intel_cache()
    if cached is not None:
        if not _should_invalidate_cache():
            return cached
        logger.info("fetch_macro_intelligence: cache invalidated by market move — refetching")

    if _get_macro_intel_client() is None:
        logger.warning("fetch_macro_intelligence: ANTHROPIC_API_KEY not set — skipping")
        return {}

    if not _macro_intel_budget_ok():
        logger.warning(
            "fetch_macro_intelligence: hourly cold-fetch budget (%d/hr) exhausted — skipping",
            MACRO_INTEL_MAX_PER_HOUR,
        )
        return {}

    today = datetime.now().strftime("%B %d, %Y")

    # key -> (query, context_prompt, max_tokens)
    searches: dict[str, tuple[str, str, int]] = {
        "lead_story": (
            f"market moving news today {today} stocks oil Fed AI earnings",
            (
                "What is the single most important development moving markets today? "
                "Search for it, then write 3-4 sentences: what happened, why it matters, "
                "and which specific tickers in the portfolio are affected and how. "
                "Be specific — name price levels, percentages, companies. "
                "Include the source in parentheses at the end."
            ),
            400,
        ),
        "central_banks": (
            f"Federal Reserve ECB BOJ rate decision {today} hike cut hold",
            (
                "What is the current central bank situation? "
                "Search and write 2-3 sentences covering: "
                "current Fed funds rate, last decision date and outcome, "
                "next meeting date, and market pricing for that meeting. "
                "If BOJ or ECB had a decision recently, include it. "
                "Specific numbers only — no ranges without evidence."
            ),
            350,
        ),
        "energy_geo": (
            f"oil price Brent WTI Saudi Arabia OPEC energy supply {today}",
            (
                "What is the oil and energy situation today? "
                "Write 2-3 sentences: current Brent and WTI prices, "
                "what is driving the move (supply/demand/geopolitical), "
                "and implications for CCJ, CEG, FCX, CAT in the portfolio. "
                "If there is a geopolitical event affecting supply, name it specifically."
            ),
            350,
        ),
        "overnight": (
            f"Nikkei Kospi Hang Seng DAX FTSE overnight {today} markets",
            (
                "Write one line per region: "
                "Asia: [Nikkei X%, Kospi X%, HSI X%] — [driver in 5 words]. "
                "Europe: [DAX X%, FTSE X%] — [driver in 5 words]. "
                "US futures: [ES X%, NQ X%] — [driver]. "
                "Use actual percentages from search results. "
                'If a market is not found, skip it rather than writing "not found".'
            ),
            250,
        ),
        "tech_ai": (
            f"AI tech Microsoft Google Apple Nvidia semiconductor {today} news",
            (
                "What happened in tech and AI today that affects the portfolio? "
                "Write 2-3 sentences covering any: earnings/guidance changes, "
                "AI narrative developments, semiconductor news, analyst actions. "
                "Name specific tickers and price reactions. "
                "If nothing material, write one sentence saying so."
            ),
            350,
        ),
        "analyst_notes": (
            f"analyst upgrade downgrade price target {_get_held_tickers()} {today}",
            (
                "List any analyst rating changes or price target changes today "
                "for stocks in the portfolio. "
                "Format each as: [Firm] [TICKER] — [action] — [old→new target] — [rationale]. "
                "Only include actions from today. "
                'If none found, write one line: "No analyst actions today."'
            ),
            300,
        ),
        "eco_releases": (
            f"economic data release today {today} CPI PPI NFP GDP retail sales actual result",
            (
                "What economic data was released today? "
                "For each release: name, actual vs consensus, beat/miss, "
                "and one-sentence market impact. "
                "If no releases today, say so in one line and name "
                "the next important release this week."
            ),
            300,
        ),
    }

    if _is_fomc_preview_week():
        searches["fomc_preview"] = (
            f"FOMC Federal Reserve meeting preview dot plot {today}",
            (
                "Write a 3-sentence FOMC preview: "
                "(1) What is the market pricing (% probability hike/hold/cut) "
                "and what changed since the last meeting? "
                "(2) What does the dot plot risk look like — "
                "one-and-done or sustained hikes? "
                "(3) What is the single most important thing to watch at 2PM ET "
                "and how does it affect TLT, MSFT, AMZN, growth multiples? "
                "Use specific numbers from search results."
            ),
            400,
        )

    results: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=len(searches)) as pool:
        future_to_key = {
            pool.submit(
                _macro_search_and_synthesize, query, context_prompt, max_tokens, regime_flags
            ): key
            for key, (query, context_prompt, max_tokens) in searches.items()
        }
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                results[key] = future.result()
            except Exception as exc:
                logger.warning("fetch_macro_intelligence: %s search failed: %s", key, exc)
                results[key] = ""

    fetched_at_iso = datetime.now().isoformat()
    _write_macro_intel_cache(results, fetched_at_iso)
    results = dict(results)
    results["_fetched_at"] = fetched_at_iso
    return results


def _is_not_found_placeholder(text: str) -> bool:
    """True only when `text` *is* the extractor's "not found"/"no ...
    today" fallback — a short whole-response placeholder — not merely
    when it *contains* that phrase.

    The extract prompts ask for one "not found" per missing *field* inside
    an otherwise multi-field markdown answer (e.g. eco_releases: "Prior:
    Not found" for one release, real data for three others) — a substring
    check there would wrongly drop the whole section. Checked by length
    (a real multi-field answer is always much longer than the placeholder
    itself) rather than an exact-string match, so a placeholder followed
    by trailing punctuation/whitespace still counts.
    """
    if not text:
        return True
    return len(text.strip()) < 40 and "not found" in text.lower()


def format_macro_intelligence(intel: dict) -> str:
    """Formats macro intelligence into brief section text.

    A section is omitted (not padded with a "nothing found" line) when its
    search found nothing or returned only the extractor's "not found"
    placeholder (see `_is_not_found_placeholder()`). `intel`'s
    "_fetched_at" key (see fetch_macro_intelligence()) is metadata, not a
    section — excluded from both the emptiness check and the section loop.
    """
    section_values = [v for k, v in intel.items() if not k.startswith("_")]
    if not any(not _is_not_found_placeholder(v) for v in section_values):
        return ""

    lines = ["\U0001F4E1 MACRO INTELLIGENCE (web-sourced)", ""]

    section_labels = {
        "lead_story": "\U0001F4CC Lead",
        "central_banks": "\U0001F3E6 Central Banks",
        "energy_geo": "⛽ Energy & Geopolitical",
        "overnight": "\U0001F319 Overnight",
        "tech_ai": "\U0001F916 Tech & AI",
        "fomc_preview": "\U0001F3DB FOMC Preview",
        "analyst_notes": "\U0001F4CB Analyst Actions",
        "eco_releases": "\U0001F4CA Today's Releases",
    }

    # No per-section character cap here — a prior 450-char cap with a
    # rsplit()-to-word-boundary-plus-"…" cut routinely sliced a section off
    # mid-sentence (the extraction pass isn't told about this limit, so it
    # writes complete sentences that this then chopped). synthesize_section()
    # and synthesize_global_signals() still cap the whole macro-intel block
    # alongside the rest of the section data before handing it to the brief
    # LLM (see their `section_data[:...]`), and send_safe() splits/sends
    # whatever's left at Telegram's message-length limit — so full section
    # content flows through here and gets truncated, if at all, at a
    # message/paragraph boundary further downstream instead of a raw
    # character count.
    for key, label in section_labels.items():
        content = intel.get(key, "")
        if not _is_not_found_placeholder(content):
            lines.append(f"*{label}*")
            lines.append(content)
            lines.append("")

    footer = "_Source: web search — verify time-sensitive facts_"
    fetched_at = intel.get("_fetched_at")
    if fetched_at:
        try:
            age_minutes = (datetime.now() - datetime.fromisoformat(fetched_at)).total_seconds() / 60
            if age_minutes < 5:
                age_str = "fresh"
            elif age_minutes < 60:
                age_str = f"{int(age_minutes)}min ago"
            else:
                age_str = f"{int(age_minutes / 60)}h ago"
            footer = f"_Fetched {age_str} — web search, verify time-sensitive facts_"
        except ValueError:
            pass
    lines.append(footer)
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    snapshot_result = fetch_market_snapshot()
    print(format_market_snapshot(snapshot_result))

    global_signals_result = fetch_global_signals()
    print()
    print(format_global_signals(global_signals_result))
