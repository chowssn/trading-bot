"""Fundamental quality scoring for a single ticker in the equity screener funnel.

Runs after `price_filter.py` has already narrowed the Russell 1000 down to a
small set of price-dislocated names. For each candidate, `score_ticker()`
pulls fundamentals from two sources:

  - yfinance (`Ticker.quarterly_cashflow`, `.quarterly_income_stmt`,
    `.quarterly_balance_sheet`, `.income_stmt`, `.cashflow`, `.info`) for
    everything except ROIC — free, no API key, but the field coverage and
    exact row-label set vary by ticker, so every fetch here is defensive
    (missing row/insufficient history degrades to `None` + a note in
    `data_warnings`, never a crash).
  - `equity.tools.roic_calculator.calculate_roic()` (FMP-backed) for ROIC.
    That function raises on insufficient FMP data rather than returning an
    error dict; `_call_roic()` below is the adapter that turns a raised
    exception into `{'roic_error': ...}` so the rest of this module can
    treat "ROIC unavailable" as a single, uniform case.

Scoring: six components (ROIC, CFO quality, leverage, share count, revenue
growth, EBITDA margin, valuation direction) sum to a 0-100 `quality_score`.
A metric that can't be computed gets a documented neutral/partial-credit
fallback (see the `_score_*` functions) rather than being treated as a
red flag — the trading thesis here is capital preservation, so we want
"we don't know" and "this is bad" to look different downstream.

`score_ticker()` never raises: any unhandled exception is caught and
reported as `tier='error'`, `quality_score=0`, with the exception message
in `data_warnings`, so one bad ticker can't take down a screener run.

Cached per-ticker to `equity/data/cache/quality_{ticker}_{date}.json` for
`settings.FUNDAMENTAL_CACHE_HOURS` hours.
"""

import json
import logging
import time
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

from equity.config import settings
from equity.tools.roic_calculator import calculate_roic

logger = logging.getLogger(__name__)

CACHE_DIR = Path(settings.CACHE_DIR)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CACHE_TTL_SECONDS = settings.FUNDAMENTAL_CACHE_HOURS * 60 * 60

# Primary + fallback row labels, in try-order, for D&A on both the quarterly
# and annual cash-flow statements. Confirmed against live yfinance output —
# 'Depreciation And Amortization' is present on both `quarterly_cashflow`
# and `cashflow`; the fallbacks cover tickers/periods where it's absent.
DA_ROW_NAMES = ["Depreciation And Amortization", "Depreciation Amortization Depletion", "Reconciled Depreciation"]

ROIC_PERIODS = 5
MUSD = 1_000_000.0  # divide raw yfinance dollar figures by this to get $M, per spec

RESULT_KEYS = [
    "ticker", "quality_score", "tier", "red_flags", "yellow_flags", "green_flags",
    "roic_current", "roic_5y_avg", "roic_available",
    "cfo_t12m", "net_income_t12m", "cfo_gte_ni",
    "ebitda_t12m", "net_debt_ebitda",
    "ebitda_margin_3y_avg", "revenue_cagr_3y", "share_count_direction",
    "forward_pe", "trailing_pe", "sector",
    "score_components", "data_warnings",
]


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------

def _cache_path(ticker: str, as_of: date | None = None) -> Path:
    as_of = as_of or date.today()
    return CACHE_DIR / f"quality_{ticker}_{as_of.isoformat()}.json"


def _is_cache_fresh(path: Path) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < CACHE_TTL_SECONDS


def _load_cached(ticker: str) -> dict | None:
    path = _cache_path(ticker)
    if not _is_cache_fresh(path):
        return None
    try:
        with path.open() as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read quality cache for %s: %s — refetching", ticker, exc)
        return None


def _json_safe(obj):
    """Recursively replace NaN floats with None so `json.dump` produces valid JSON."""
    if isinstance(obj, float) and obj != obj:  # NaN != NaN
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def _save_cache(ticker: str, result: dict) -> None:
    path = _cache_path(ticker)
    try:
        with path.open("w") as f:
            json.dump(_json_safe(result), f)
    except OSError as exc:
        logger.warning("Failed to write quality cache for %s: %s", ticker, exc)


# --------------------------------------------------------------------------
# Statement helpers — all defensive: missing row / short history -> None,
# never a raised exception.
# --------------------------------------------------------------------------

def _safe_float(value) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if f != f else f  # reject NaN


def _get_row(df: pd.DataFrame | None, row_names: list[str]) -> pd.Series | None:
    """First matching row (already most-recent-first, per yfinance convention) or None."""
    if df is None or df.empty:
        return None
    for name in row_names:
        if name in df.index:
            return df.loc[name]
    return None


def _sum_last_n(row: pd.Series | None, n: int) -> float | None:
    """Sum of the first `n` non-null values of `row` (most-recent-first), or None."""
    if row is None:
        return None
    vals = row.dropna()
    if vals.empty:
        return None
    return float(vals.iloc[:n].sum())


def _first_value(row: pd.Series | None) -> float | None:
    if row is None:
        return None
    vals = row.dropna()
    return float(vals.iloc[0]) if not vals.empty else None


def _cagr_3y(row: pd.Series | None) -> float | None:
    """3-year CAGR (%) from a most-recent-first annual row: (latest/3y_ago)^(1/3) - 1) * 100.

    None if fewer than 4 annual periods are available, or if either endpoint
    is non-positive (avoids a complex/undefined result).
    """
    if row is None:
        return None
    vals = row.dropna()
    if len(vals) < 4:
        return None
    latest, base = float(vals.iloc[0]), float(vals.iloc[3])
    if latest <= 0 or base <= 0:
        return None
    return ((latest / base) ** (1 / 3) - 1) * 100


def _ebitda_margin_3y_avg(op_income_row, da_row, revenue_row) -> tuple[float | None, str | None]:
    """Average of (Op Income + D&A) / Revenue * 100 over up to the last 3 common annual periods."""
    if op_income_row is None or da_row is None or revenue_row is None:
        return None, "ebitda_margin_3y_avg unavailable: missing annual Operating Income/D&A/Revenue"

    common_dates = sorted(
        set(op_income_row.dropna().index) & set(da_row.dropna().index) & set(revenue_row.dropna().index),
        reverse=True,
    )[:3]
    if not common_dates:
        return None, "ebitda_margin_3y_avg unavailable: no overlapping annual periods across statements"

    margins = []
    for d in common_dates:
        revenue = revenue_row[d]
        if not revenue:
            continue
        margins.append((op_income_row[d] + da_row[d]) / revenue * 100)

    if not margins:
        return None, "ebitda_margin_3y_avg unavailable: zero/missing revenue in all overlapping periods"

    warning = None if len(margins) >= 3 else f"ebitda_margin_3y_avg computed from only {len(margins)} annual period(s)"
    return sum(margins) / len(margins), warning


def _fetch_quarterly_metrics(tk: "yf.Ticker", warnings: list[str]) -> dict:
    qcf = getattr(tk, "quarterly_cashflow", None)
    qis = getattr(tk, "quarterly_income_stmt", None)
    qbs = getattr(tk, "quarterly_balance_sheet", None)

    cfo_t12m = _sum_last_n(_get_row(qcf, ["Operating Cash Flow"]), 4)
    net_income_t12m = _sum_last_n(_get_row(qis, ["Net Income"]), 4)
    op_income_t12m = _sum_last_n(_get_row(qis, ["Operating Income"]), 4)
    da_t12m = _sum_last_n(_get_row(qcf, DA_ROW_NAMES), 4)

    if cfo_t12m is None:
        warnings.append("cfo_t12m unavailable: no 'Operating Cash Flow' row in quarterly_cashflow")
    else:
        cfo_t12m /= MUSD
    if net_income_t12m is None:
        warnings.append("net_income_t12m unavailable: no 'Net Income' row in quarterly_income_stmt")
    else:
        net_income_t12m /= MUSD

    ebitda_t12m = None
    if op_income_t12m is not None and da_t12m is not None:
        ebitda_t12m = (op_income_t12m + da_t12m) / MUSD
    else:
        warnings.append("ebitda_t12m unavailable: missing quarterly Operating Income and/or D&A")

    net_debt = _first_value(_get_row(qbs, ["Net Debt"]))
    if net_debt is None:
        warnings.append("net_debt unavailable: no 'Net Debt' row in quarterly_balance_sheet")
    else:
        net_debt /= MUSD

    net_debt_ebitda = None
    if net_debt is not None and ebitda_t12m:
        net_debt_ebitda = max(-10.0, min(20.0, net_debt / ebitda_t12m))
    elif net_debt is not None and ebitda_t12m is not None and not ebitda_t12m:
        warnings.append("net_debt_ebitda unavailable: ebitda_t12m is zero")

    return {
        "cfo_t12m": cfo_t12m,
        "net_income_t12m": net_income_t12m,
        "cfo_gte_ni": (cfo_t12m is not None and net_income_t12m is not None and cfo_t12m >= net_income_t12m),
        "ebitda_t12m": ebitda_t12m,
        "net_debt": net_debt,
        "net_debt_ebitda": net_debt_ebitda,
    }


def _fetch_annual_metrics(tk: "yf.Ticker", warnings: list[str]) -> dict:
    inc = getattr(tk, "income_stmt", None)
    cf = getattr(tk, "cashflow", None)

    revenue_row = _get_row(inc, ["Total Revenue"])
    op_income_row = _get_row(inc, ["Operating Income"])
    da_row = _get_row(cf, DA_ROW_NAMES)
    shares_row = _get_row(inc, ["Basic Average Shares"])

    margin, margin_warning = _ebitda_margin_3y_avg(op_income_row, da_row, revenue_row)
    if margin_warning:
        warnings.append(margin_warning)

    revenue_cagr = _cagr_3y(revenue_row)
    if revenue_cagr is None:
        warnings.append("revenue_cagr_3y unavailable: fewer than 4 annual 'Total Revenue' periods")

    share_cagr = _cagr_3y(shares_row)
    if share_cagr is None:
        warnings.append("share_count_direction unavailable: fewer than 4 annual 'Basic Average Shares' periods — defaulting to neutral")
        share_direction = "neutral"
    elif share_cagr < 0:
        share_direction = "buyback"
    elif share_cagr > 2:
        share_direction = "dilutive"
    else:
        share_direction = "neutral"

    return {
        "ebitda_margin_3y_avg": margin,
        "revenue_cagr_3y": revenue_cagr,
        "share_count_direction": share_direction,
        "share_count_cagr": share_cagr,  # internal only — not part of the public result schema
    }


def _fetch_info_metrics(tk: "yf.Ticker", warnings: list[str]) -> dict:
    try:
        info = tk.info or {}
    except Exception as exc:  # yfinance can raise a variety of things on network/format issues
        logger.warning("yfinance .info fetch failed: %s", exc)
        warnings.append(f"info unavailable: {exc}")
        info = {}
    return {
        "forward_pe": _safe_float(info.get("forwardPE")),
        "trailing_pe": _safe_float(info.get("trailingPE")),
        "sector": info.get("sector"),
    }


def _call_roic(ticker: str, api_key: str, warnings: list[str]) -> dict:
    """Adapter: `calculate_roic()` raises on insufficient data; normalize that to `{'roic_error': ...}`."""
    try:
        return calculate_roic(ticker, api_key, periods=ROIC_PERIODS)
    except Exception as exc:
        logger.warning("ROIC unavailable for %s: %s", ticker, exc)
        warnings.append(f"roic unavailable: {exc}")
        return {"roic_current": None, "roic_5y_avg": None, "roic_error": str(exc)}


# --------------------------------------------------------------------------
# Scoring — each _score_* returns (points, red_flags, yellow_flags, green_flags)
# --------------------------------------------------------------------------

def _score_roic(roic_current: float | None, roic_available: bool) -> tuple[int, list, list, list]:
    if not roic_available or roic_current is None:
        return 15, [], ["roic_unverified"], []
    if roic_current > 25:
        return 25, [], [], []
    if roic_current > 20:
        return 22, [], [], []
    if roic_current > 15:
        return 18, [], [], []
    if roic_current > 10:
        return 10, [], [], []
    return 0, ["low_roic"], [], []


def _score_cfo_quality(cfo_t12m: float | None, net_income_t12m: float | None, cfo_gte_ni: bool) -> tuple[int, list, list, list]:
    if cfo_gte_ni:
        return 15, [], [], []
    if cfo_t12m is None or not net_income_t12m or net_income_t12m <= 0:
        return 0, [], ["weak_cfo"], []
    ratio = cfo_t12m / net_income_t12m
    if ratio >= 0.8:
        return 12, [], [], []
    if ratio >= 0.6:
        return 8, [], [], []
    return 0, [], ["weak_cfo"], []


def _score_leverage(net_debt: float | None, net_debt_ebitda: float | None) -> tuple[int, list, list, list]:
    if net_debt is not None and net_debt < 0:
        return 15, [], [], ["net_cash"]
    if net_debt_ebitda is None:
        return 5, [], ["leverage_data_unavailable"], []
    if net_debt_ebitda < 1:
        return 15, [], [], []
    if net_debt_ebitda < 2:
        return 12, [], [], []
    if net_debt_ebitda < 3:
        return 8, [], [], []
    if net_debt_ebitda < 4:
        return 3, [], ["elevated_leverage"], []
    return 0, ["high_leverage"], [], []


def _score_share_count(direction: str, share_cagr: float | None) -> tuple[int, list, list, list]:
    if direction == "buyback":
        return 10, [], [], []
    if direction == "neutral":
        return 8, [], [], []
    # dilutive
    if share_cagr is not None and share_cagr > 5:
        return 0, ["heavy_dilution"], [], []
    return 4, [], ["dilutive"], []


def _score_revenue_growth(cagr: float | None) -> tuple[int, list, list, list]:
    if cagr is None:
        return 5, [], ["revenue_data_unavailable"], []
    if cagr > 15:
        return 15, [], [], []
    if cagr > 10:
        return 12, [], [], []
    if cagr > 5:
        return 9, [], [], []
    if cagr > 0:
        return 5, [], [], []
    return 0, [], ["declining_revenue"], []


def _score_ebitda_margin(roic_current: float | None, roic_available: bool, margin: float | None) -> tuple[int, list, list, list]:
    if margin is None or roic_current is None or not roic_available:
        return 4, [], [], []
    if roic_current > 25:
        if margin >= 15:
            return 10, [], [], []
        if margin >= 10:
            return 8, [], [], []
        return 5, [], [], []
    if roic_current >= 15:
        if margin >= 20:
            return 10, [], [], []
        if margin >= 15:
            return 8, [], [], []
        return 4, [], [], []
    # roic < 15%
    if margin >= 25:
        return 5, [], [], []
    if margin >= 15:
        # Gap in the spec (only "roic<15 and margin<15" is documented) — fill
        # conservatively, below every roic>=15 tier's output.
        return 2, [], [], []
    return 0, [], ["poor_margin_low_roic"], []


def _score_valuation(forward_pe: float | None, trailing_pe: float | None) -> tuple[int, list, list, list]:
    if forward_pe is None or not trailing_pe:
        return 5, [], [], []
    if forward_pe < trailing_pe:
        return 10, [], [], []
    if abs(forward_pe - trailing_pe) <= 0.10 * abs(trailing_pe):
        return 6, [], [], []
    return 2, [], ["pe_expansion"], []


def _tier_for(score: int) -> str:
    if score >= 70:
        return "tier1"
    if score >= 50:
        return "tier2"
    if score >= 30:
        return "tier3"
    return "below_threshold"


def _empty_result(ticker: str) -> dict:
    """Full result schema with neutral/empty defaults — base for both success and error paths."""
    return {
        "ticker": ticker,
        "quality_score": 0,
        "tier": "below_threshold",
        "red_flags": [],
        "yellow_flags": [],
        "green_flags": [],
        "roic_current": None,
        "roic_5y_avg": None,
        "roic_available": False,
        "cfo_t12m": None,
        "net_income_t12m": None,
        "cfo_gte_ni": False,
        "ebitda_t12m": None,
        "net_debt_ebitda": None,
        "ebitda_margin_3y_avg": None,
        "revenue_cagr_3y": None,
        "share_count_direction": "neutral",
        "forward_pe": None,
        "trailing_pe": None,
        "sector": None,
        "score_components": {},
        "data_warnings": [],
    }


def _score_ticker_uncached(ticker: str, api_key: str) -> dict:
    warnings: list[str] = []
    tk = yf.Ticker(ticker)

    quarterly = _fetch_quarterly_metrics(tk, warnings)
    annual = _fetch_annual_metrics(tk, warnings)
    info_metrics = _fetch_info_metrics(tk, warnings)
    roic_result = _call_roic(ticker, api_key, warnings)

    roic_current = roic_result.get("roic_current")
    roic_5y_avg = roic_result.get("roic_5y_avg")
    roic_available = "roic_error" not in roic_result

    components = {}
    red_flags, yellow_flags, green_flags = [], [], []

    def _add(name: str, points: int, red: list, yellow: list, green: list):
        components[name] = points
        red_flags.extend(red)
        yellow_flags.extend(yellow)
        green_flags.extend(green)

    _add("roic", *_score_roic(roic_current, roic_available))
    _add("cfo_quality", *_score_cfo_quality(quarterly["cfo_t12m"], quarterly["net_income_t12m"], quarterly["cfo_gte_ni"]))
    _add("leverage", *_score_leverage(quarterly["net_debt"], quarterly["net_debt_ebitda"]))
    _add("share_count", *_score_share_count(annual["share_count_direction"], annual["share_count_cagr"]))
    _add("revenue_growth", *_score_revenue_growth(annual["revenue_cagr_3y"]))
    _add("ebitda_margin", *_score_ebitda_margin(roic_current, roic_available, annual["ebitda_margin_3y_avg"]))
    _add("valuation", *_score_valuation(info_metrics["forward_pe"], info_metrics["trailing_pe"]))

    quality_score = sum(components.values())

    result = _empty_result(ticker)
    result.update({
        "quality_score": quality_score,
        "tier": _tier_for(quality_score),
        "red_flags": red_flags,
        "yellow_flags": yellow_flags,
        "green_flags": green_flags,
        "roic_current": roic_current,
        "roic_5y_avg": roic_5y_avg,
        "roic_available": roic_available,
        "cfo_t12m": quarterly["cfo_t12m"],
        "net_income_t12m": quarterly["net_income_t12m"],
        "cfo_gte_ni": quarterly["cfo_gte_ni"],
        "ebitda_t12m": quarterly["ebitda_t12m"],
        "net_debt_ebitda": quarterly["net_debt_ebitda"],
        "ebitda_margin_3y_avg": annual["ebitda_margin_3y_avg"],
        "revenue_cagr_3y": annual["revenue_cagr_3y"],
        "share_count_direction": annual["share_count_direction"],
        "forward_pe": info_metrics["forward_pe"],
        "trailing_pe": info_metrics["trailing_pe"],
        "sector": info_metrics["sector"],
        "score_components": components,
        "data_warnings": warnings,
    })
    return result


def score_ticker(ticker: str, api_key: str) -> dict:
    """Fetch fundamentals for `ticker` and return a quality score dict (see module docstring).

    Never raises — any unhandled exception is caught and returned as
    `{'quality_score': 0, 'tier': 'error', 'data_warnings': [...]}`.
    Cached to `equity/data/cache/quality_{ticker}_{date}.json` for
    `settings.FUNDAMENTAL_CACHE_HOURS` hours.
    """
    try:
        cached = _load_cached(ticker)
        if cached is not None:
            return cached

        result = _score_ticker_uncached(ticker, api_key)
        _save_cache(ticker, result)
        return result
    except Exception as exc:
        logger.exception("Unhandled error scoring %s", ticker)
        result = _empty_result(ticker)
        result["tier"] = "error"
        result["data_warnings"] = [str(exc)]
        return result
