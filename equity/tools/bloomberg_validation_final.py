"""Consolidated Bloomberg field validation for the equity screener.

Documents, in one self-contained script, the confirmed data-source mapping
for every field the screener consumes: which source produces it, how well
it agrees with Bloomberg Terminal, and how the screener should treat it
(hard filter vs. directional signal vs. approved-with-caveat).

Two data sources, used together:

    yfinance  -- primary source for price data, financial statements, and
                 ratios (free, no key required).
    FMP free tier ("/stable" endpoints, FMP_API_KEY from .env) -- ROIC
                 custom calculation only, via `roic_calculator.calculate_roic`.

This replaces `bloomberg_validation.py`, `bloomberg_validation_v2.py`, and
any v3 -- do not import from those; this script is self-contained and is
the one to keep updated going forward. It imports only from
`equity/tools/roic_calculator.py` and `backtest/indicators.py`.

Bloomberg reference values (`MSFT_BBG`, `ADBE_BBG` below) are hardcoded --
this is a point-in-time validation snapshot, not a live lookup. Re-pull
from the Terminal and update them if you re-run this after a new fiscal
period closes.

Usage: python -m equity.tools.bloomberg_validation_final

Output: console report (per-ticker field-by-field, then a cross-ticker
summary table and the data-source decision) plus
`equity/data/bloomberg_validation_final.csv` (one row per field per ticker).
"""

import csv
import logging
import os
from pathlib import Path

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

from backtest.indicators import rsi
from equity.tools.roic_calculator import calculate_roic

logger = logging.getLogger(__name__)

CSV_OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "bloomberg_validation_final.csv"

# --- Bloomberg Terminal reference values, hardcoded for validation ---

MSFT_BBG = {
    "roic_current": 25.02,
    "roic_5y_avg": 28.11,
    "cfo_t12m": 182935.0,
    "net_income_t12m": 133749.0,
    "ebitda_t12m": 200739.0,
    "ebitda_margin_3y_avg": 58.46,
    "net_debt": 51970.0,
    "net_debt_ebitda": 0.259,
    "trailing_pe": 29.12,
    "forward_pe": 20.547,
    "return_1y": -3.9413,
    "rsi_14d": 62.419,
    "rsi_30d": 60.890,
    "share_count_direction": "buyback",
    "share_count_5y_geo": -0.31,
}

ADBE_BBG = {
    "roic_current": 38.92,
    "roic_5y_avg": 27.25,
    "cfo_t12m": 10481.0,
    "net_income_t12m": 7130.0,
    "ebitda_t12m": 9849.0,
    "ebitda_margin_3y_avg": 38.55,
    "net_debt": 53.0,
    "net_debt_ebitda": 0.0054,
    "trailing_pe": 15.49,
    "forward_pe": 10.043,
    "return_1y": -23.9692,
    "rsi_14d": 62.514,
    "rsi_30d": 58.232,
    "share_count_direction": "buyback",
    "share_count_5y_geo": -2.40,
}

TICKERS_BBG = {"MSFT": MSFT_BBG, "ADBE": ADBE_BBG}

# --- Deviation thresholds ---
# ROIC and EBITDA margin are both dimensionless percentages of the same order
# of magnitude, so margin reuses the ROIC pp thresholds (no separate margin
# threshold is specified against Bloomberg).
ROIC_GREEN_PP = 3.0
ROIC_YELLOW_PP = 8.0
MARGIN_GREEN_PP = ROIC_GREEN_PP
MARGIN_YELLOW_PP = ROIC_YELLOW_PP

MONETARY_GREEN_PCT = 2.0
MONETARY_YELLOW_PCT = 5.0

RETURN_GREEN_PCT = 2.0
RETURN_YELLOW_PCT = 10.0

# Directional ratio fields (P/E) have no Bloomberg-defined agreement band;
# "PASS/FAIL agreement" is implemented here as: does our value land within
# 15% of Bloomberg's -- close enough to draw the same cheap/expensive read.
PE_AGREEMENT_PCT = 15.0

NET_DEBT_EBITDA_TIERS = (
    (1.0, "excellent"),
    (2.0, "good"),
    (3.0, "flag"),
)

_ANSI_GREEN = "\033[92m"
_ANSI_YELLOW = "\033[93m"
_ANSI_RED = "\033[91m"
_ANSI_CYAN = "\033[96m"
_ANSI_RESET = "\033[0m"

_ANSI_BY_FLAG = {
    "GREEN": _ANSI_GREEN,
    "YELLOW": _ANSI_YELLOW,
    "RED": _ANSI_RED,
    "DIRECTIONAL": _ANSI_CYAN,
    "APPROVED_WITH_CAVEAT": _ANSI_CYAN,
    "N/A": _ANSI_RESET,
}


def _colorize(flag: str) -> str:
    return f"{_ANSI_BY_FLAG.get(flag, _ANSI_RESET)}{flag}{_ANSI_RESET}"


# ============================================================
# Field fetchers
# ============================================================

def _row(df: pd.DataFrame, names) -> pd.Series | None:
    """First matching row from `df.loc`, trying each name in `names` in order.

    `names` may be a single row name or a list of fallback names. Returns
    None if the frame is missing or none of the names are present, rather
    than raising KeyError.
    """
    if df is None or df.empty:
        return None
    if isinstance(names, str):
        names = [names]
    for name in names:
        if name in df.index:
            return df.loc[name]
    return None


def fetch_roic(ticker: str, api_key: str) -> dict:
    """FMP custom ROIC calculation, via `roic_calculator.calculate_roic`."""
    try:
        result = calculate_roic(ticker, api_key, periods=5)
    except (RuntimeError, ValueError) as exc:
        # calculate_roic surfaces FMP access/subscription problems (402/403/404,
        # or a tier that caps history below what's needed) as RuntimeError once
        # its statement fetches come back empty -- it doesn't preserve the HTTP
        # status, so we treat any failure here as an FMP access problem.
        logger.warning("ROIC calculation failed for %s: %s", ticker, exc)
        return {"roic_current": None, "roic_5y_avg": None, "roic_error": "FMP subscription required"}
    return {"roic_current": result["roic_current"], "roic_5y_avg": result["roic_5y_avg"]}


def fetch_cashflow_fields(ticker: str) -> dict:
    """CFO, Net Income, EBITDA (T12M) from yfinance quarterly statements."""
    t = yf.Ticker(ticker)
    qcf = t.quarterly_cashflow
    qis = t.quarterly_income_stmt

    cfo_row = _row(qcf, "Operating Cash Flow")
    ni_row = _row(qis, "Net Income")
    op_income_row = _row(qis, "Operating Income")
    da_row = _row(qcf, ["Depreciation And Amortization", "Depreciation Amortization Depletion", "Reconciled Depreciation"])

    cfo_t12m = float(cfo_row.iloc[:4].sum()) / 1e6 if cfo_row is not None else None
    net_income_t12m = float(ni_row.iloc[:4].sum()) / 1e6 if ni_row is not None else None
    cfo_gte_ni = cfo_t12m >= net_income_t12m if cfo_t12m is not None and net_income_t12m is not None else None

    ebitda_t12m = None
    if op_income_row is not None and da_row is not None:
        ebitda_t12m = (float(op_income_row.iloc[:4].sum()) + float(da_row.iloc[:4].sum())) / 1e6

    return {
        "cfo_t12m": cfo_t12m,
        "net_income_t12m": net_income_t12m,
        "cfo_gte_ni": cfo_gte_ni,
        "ebitda_t12m": ebitda_t12m,
    }


def fetch_margin_fields(ticker: str) -> dict:
    """3Y average EBITDA margin from yfinance annual financials.

    EBITDA per period is Operating Income + D&A (same method as
    `fetch_cashflow_fields`), not yfinance's pre-calculated EBITDA row.
    """
    t = yf.Ticker(ticker)
    inc = t.income_stmt  # annual, most-recent-first columns
    cf = t.cashflow  # annual

    revenue_row = _row(inc, "Total Revenue")
    op_income_row = _row(inc, "Operating Income")
    da_row = _row(cf, ["Depreciation And Amortization", "Depreciation Amortization Depletion", "Reconciled Depreciation"])

    if revenue_row is None or op_income_row is None or da_row is None:
        return {"ebitda_margin_3y_avg": None, "ebitda_margin_years_used": 0}

    n_periods = min(3, len(revenue_row), len(op_income_row), len(da_row))
    margins = []
    for i in range(n_periods):
        revenue = revenue_row.iloc[i]
        op_income = op_income_row.iloc[i]
        da = da_row.iloc[i]
        if pd.isna(revenue) or not revenue or pd.isna(op_income) or pd.isna(da):
            continue
        ebitda = op_income + da
        margins.append((ebitda / revenue) * 100)

    ebitda_margin_3y_avg = sum(margins) / len(margins) if margins else None
    return {"ebitda_margin_3y_avg": ebitda_margin_3y_avg, "ebitda_margin_years_used": len(margins)}


def _classify_net_debt_tier(net_debt_ebitda_ratio: float | None) -> str | None:
    """excellent (<1x) / good (1-2x) / flag (2-3x) / exclude (>3x)."""
    if net_debt_ebitda_ratio is None:
        return None
    for ceiling, tier in NET_DEBT_EBITDA_TIERS:
        if net_debt_ebitda_ratio < ceiling:
            return tier
    return "exclude"


def fetch_balance_sheet_fields(ticker: str, ebitda_t12m: float | None) -> dict:
    """Net debt (and tiered Net Debt/EBITDA) from yfinance quarterly balance sheet.

    yfinance's Net Debt deviates from Bloomberg's due to financial-subsidiary
    adjustments Bloomberg makes that yfinance does not -- screen on the
    tiered ratio, not the absolute value.
    """
    t = yf.Ticker(ticker)
    qbs = t.quarterly_balance_sheet

    net_debt_row = _row(qbs, "Net Debt")
    if net_debt_row is not None and not pd.isna(net_debt_row.iloc[0]):
        net_debt = float(net_debt_row.iloc[0]) / 1e6
    else:
        total_debt_row = _row(qbs, "Total Debt")
        cash_row = _row(qbs, "Cash And Cash Equivalents")
        net_debt = (
            (float(total_debt_row.iloc[0]) - float(cash_row.iloc[0])) / 1e6
            if total_debt_row is not None and cash_row is not None
            else None
        )

    net_debt_ebitda_ratio = net_debt / ebitda_t12m if net_debt is not None and ebitda_t12m else None

    return {
        "net_debt": net_debt,
        "net_debt_ebitda_ratio": net_debt_ebitda_ratio,
        "net_debt_tier": _classify_net_debt_tier(net_debt_ebitda_ratio),
    }


def fetch_valuation_fields(ticker: str) -> dict:
    """Trailing/forward P/E from yfinance `.info` -- directional signals only."""
    info = yf.Ticker(ticker).info
    return {"trailing_pe": info.get("trailingPE"), "forward_pe": info.get("forwardPE")}


def _return_over_calendar_years(history: pd.DataFrame, years: int) -> float | None:
    """% price return from ~`years` calendar years ago to the most recent close.

    Finds the closest trading day to exactly `years` calendar years before
    the most recent bar via `index.searchsorted()`.
    """
    if history.empty:
        return None
    price_today = history["Close"].iloc[-1]
    target_date = history.index[-1] - pd.DateOffset(years=years)
    pos = min(history.index.searchsorted(target_date), len(history) - 1)
    price_then = history["Close"].iloc[pos]
    if pd.isna(price_today) or pd.isna(price_then) or not price_then:
        return None
    return (price_today / price_then - 1) * 100


def fetch_price_fields(ticker: str) -> dict:
    """1Y/5Y return and RSI(14/30) from yfinance daily history."""
    t = yf.Ticker(ticker)

    hist_1y = t.history(period="2y", auto_adjust=True)  # 2y buffer for a 1y-back lookup
    hist_5y = t.history(period="6y", auto_adjust=True)  # 6y buffer for a 5y-back lookup

    return_1y = _return_over_calendar_years(hist_1y, 1)
    return_5y = _return_over_calendar_years(hist_5y, 5)

    close = hist_1y["Close"]
    rsi_14d_series = rsi(close, period=14)
    rsi_30d_series = rsi(close, period=30)
    rsi_14d = float(rsi_14d_series.iloc[-1]) if not close.empty and not pd.isna(rsi_14d_series.iloc[-1]) else None
    rsi_30d = float(rsi_30d_series.iloc[-1]) if not close.empty and not pd.isna(rsi_30d_series.iloc[-1]) else None

    return {"return_1y": return_1y, "return_5y": return_5y, "rsi_14d": rsi_14d, "rsi_30d": rsi_30d}


def fetch_share_count_fields(ticker: str) -> dict:
    """Share-count CAGR/direction from yfinance annual income statement.

    Bloomberg uses a 5Y geometric average of basic shares; yfinance's annual
    income statement only exposes 3-4 years, so this is a 3Y (4 data points)
    or 4Y (5 data points) CAGR -- a shorter window than Bloomberg's.
    """
    t = yf.Ticker(ticker)
    inc = t.income_stmt  # annual, most-recent-first columns

    shares_row = _row(inc, ["Basic Average Shares", "Diluted Average Shares"])
    if shares_row is None:
        return {"share_count_cagr": None, "share_count_direction": None, "share_count_years": 0}

    valid = shares_row.dropna()
    if len(valid) < 2:
        return {"share_count_cagr": None, "share_count_direction": None, "share_count_years": 0}

    shares_now = float(valid.iloc[0])
    shares_then = float(valid.iloc[-1])
    n_years = len(valid) - 1  # 4 data points -> 3Y CAGR, 5 data points -> 4Y CAGR

    cagr_pct = None
    if shares_now > 0 and shares_then > 0:
        cagr_pct = ((shares_now / shares_then) ** (1 / n_years) - 1) * 100

    if cagr_pct is None:
        direction = None
    elif cagr_pct < 0:
        direction = "buyback"
    elif cagr_pct > 2:
        direction = "dilutive"
    else:
        direction = "neutral"

    return {"share_count_cagr": cagr_pct, "share_count_direction": direction, "share_count_years": n_years}


# ============================================================
# Field metadata + comparison
# ============================================================

# category -> how a field is scored against Bloomberg.
#   roic / margin:      GREEN/YELLOW/RED on absolute pp deviation
#   monetary:            GREEN/YELLOW/RED on relative % deviation
#   return_rsi:          GREEN/YELLOW/RED on relative % deviation, pp gap also recorded
#   pe / net_debt_ebitda: DIRECTIONAL, PASS/FAIL agreement noted in caveat
#   net_debt_absolute:    APPROVED_WITH_CAVEAT (use the tiered ratio instead)
#   share_count_direction/cagr: DIRECTIONAL, PASS/FAIL agreement noted in caveat
FIELD_SPECS = [
    {
        "key": "roic_current", "ours_key": "roic_current", "label": "ROIC Current",
        "source": "FMP", "category": "roic",
        "screening_use": "Primary quality sort",
        "caveat": "Custom calc via NOPAT/AvgIC. Omits pension/FX adj (unavailable in FMP).",
    },
    {
        "key": "roic_5y_avg", "ours_key": "roic_5y_avg", "label": "ROIC 5Y Avg",
        "source": "FMP", "category": "roic",
        "screening_use": "Quality trend / consistency filter",
        "caveat": "Simple average of annual ROIC, not T12M-weighted (see roic_calculator methodology_notes).",
    },
    {
        "key": "cfo_t12m", "ours_key": "cfo_t12m", "label": "CFO T12M",
        "source": "YF", "category": "monetary",
        "screening_use": "CFO >= Net Income earnings quality check",
        "caveat": None,
    },
    {
        "key": "net_income_t12m", "ours_key": "net_income_t12m", "label": "Net Income T12M",
        "source": "YF", "category": "monetary",
        "screening_use": "CFO >= Net Income earnings quality check; profitability filter",
        "caveat": None,
    },
    {
        "key": "ebitda_t12m", "ours_key": "ebitda_t12m", "label": "EBITDA T12M",
        "source": "YF", "category": "monetary",
        "screening_use": "Net Debt/EBITDA leverage ratio denominator",
        "caveat": "Op Income + D&A method, not yfinance's pre-calculated EBITDA row.",
    },
    {
        "key": "ebitda_margin_3y_avg", "ours_key": "ebitda_margin_3y_avg", "label": "EBITDA Margin 3Y Avg",
        "source": "YF", "category": "margin",
        "screening_use": "Margin quality / stability filter",
        "caveat": "3Y average (not Bloomberg's window); Op Income + D&A method.",
    },
    {
        "key": "net_debt", "ours_key": "net_debt", "label": "Net Debt",
        "source": "YF", "category": "net_debt_absolute",
        "screening_use": "Input to Net Debt/EBITDA leverage ratio only -- not screened on directly",
        "caveat": "Deviates from Bloomberg due to financial-subsidiary adjustments. Use the tiered ratio, not the absolute value.",
    },
    {
        "key": "net_debt_ebitda", "ours_key": "net_debt_ebitda_ratio", "label": "Net Debt/EBITDA",
        "source": "YF", "category": "net_debt_ebitda",
        "screening_use": "Leverage tier classification (excellent/good/flag/exclude)",
        "caveat": "Directional only -- see Net Debt caveat.",
    },
    {
        "key": "trailing_pe", "ours_key": "trailing_pe", "label": "Trailing P/E",
        "source": "YF", "category": "pe",
        "screening_use": "Valuation context, not a hard screen",
        "caveat": "Directional signal only.",
    },
    {
        "key": "forward_pe", "ours_key": "forward_pe", "label": "Forward P/E",
        "source": "YF", "category": "pe",
        "screening_use": "Valuation context, not a hard screen",
        "caveat": "Directional signal only; consensus-estimate composition may differ from Bloomberg's.",
    },
    {
        "key": "return_1y", "ours_key": "return_1y", "label": "1Y Return",
        "source": "YF", "category": "return_rsi",
        "screening_use": "Momentum sleeve input",
        "caveat": None,
    },
    {
        "key": "rsi_14d", "ours_key": "rsi_14d", "label": "RSI 14D",
        "source": "YF", "category": "return_rsi",
        "screening_use": "Technical overbought/oversold filter",
        "caveat": None,
    },
    {
        "key": "rsi_30d", "ours_key": "rsi_30d", "label": "RSI 30D",
        "source": "YF", "category": "return_rsi",
        "screening_use": "Technical overbought/oversold filter (slower)",
        "caveat": None,
    },
    {
        "key": "share_count_direction", "ours_key": "share_count_direction", "label": "Share Count Direction",
        "source": "YF", "category": "share_count_direction",
        "screening_use": "Buyback/dilution quality filter",
        "caveat": "Direction agreement only.",
    },
    {
        "key": "share_count_5y_geo", "ours_key": "share_count_cagr", "label": "Share Count CAGR",
        "source": "YF", "category": "share_count_cagr",
        "screening_use": "Buyback/dilution magnitude context",
        "caveat": "yfinance limited to 3-4Y annual periods vs. Bloomberg's 5Y geometric average.",
    },
]


def _pct_dev(ours, bbg) -> float | None:
    if ours is None or bbg is None or not bbg:
        return None
    return abs(ours - bbg) / abs(bbg) * 100


def _pp_dev(ours, bbg) -> float | None:
    if ours is None or bbg is None:
        return None
    return abs(ours - bbg)


def _threshold_flag(dev: float | None, green_max: float, yellow_max: float) -> str:
    if dev is None:
        return "N/A"
    if dev <= green_max:
        return "GREEN"
    if dev <= yellow_max:
        return "YELLOW"
    return "RED"


def _evaluate_field(spec: dict, value_ours, value_bbg) -> dict:
    category = spec["category"]
    caveat = spec["caveat"]

    # share_count_direction compares strings ('buyback'/'dilutive'/'neutral'),
    # not numbers -- skip the numeric deviation helpers for it.
    if category == "share_count_direction":
        dev_pct = dev_pp = None
    else:
        dev_pct = _pct_dev(value_ours, value_bbg)
        dev_pp = _pp_dev(value_ours, value_bbg)

    if category in ("roic", "margin"):
        green, yellow = (ROIC_GREEN_PP, ROIC_YELLOW_PP) if category == "roic" else (MARGIN_GREEN_PP, MARGIN_YELLOW_PP)
        flag = _threshold_flag(dev_pp, green, yellow)
        dev_pct = None  # ROIC/margin are scored in pp, not relative %

    elif category == "monetary":
        flag = _threshold_flag(dev_pct, MONETARY_GREEN_PCT, MONETARY_YELLOW_PCT)
        dev_pp = None  # $M-scale fields aren't meaningfully compared in pp

    elif category == "return_rsi":
        flag = _threshold_flag(dev_pct, RETURN_GREEN_PCT, RETURN_YELLOW_PCT)
        # dev_pp (absolute point gap) kept alongside for context, per spec.

    elif category == "pe":
        flag = "DIRECTIONAL"
        if dev_pct is None:
            agreement = "N/A"
        else:
            agreement = "PASS" if dev_pct < PE_AGREEMENT_PCT else "FAIL"
        caveat = f"{caveat} Agreement (within {PE_AGREEMENT_PCT:.0f}%): {agreement}."

    elif category == "net_debt_ebitda":
        flag = "DIRECTIONAL"
        tier_ours = _classify_net_debt_tier(value_ours)
        tier_bbg = _classify_net_debt_tier(value_bbg)
        agreement = "PASS" if tier_ours is not None and tier_ours == tier_bbg else "FAIL"
        caveat = f"{caveat} Tier ours={tier_ours or 'N/A'} bbg={tier_bbg or 'N/A'} -- Agreement: {agreement}."

    elif category == "net_debt_absolute":
        flag = "APPROVED_WITH_CAVEAT"

    elif category == "share_count_direction":
        flag = "DIRECTIONAL"
        if value_ours is None or value_bbg is None:
            agreement = "N/A"
        else:
            agreement = "PASS" if value_ours == value_bbg else "FAIL"
        caveat = f"{caveat} Agreement: {agreement}."

    elif category == "share_count_cagr":
        flag = "DIRECTIONAL"
        dev_pct = None
        if value_ours is None or value_bbg is None:
            agreement = "N/A"
        else:
            agreement = "PASS" if (value_ours < 0) == (value_bbg < 0) else "FAIL"
        caveat = f"{caveat} Sign agreement: {agreement}."

    else:  # pragma: no cover - exhaustive over categories defined above
        raise ValueError(f"Unknown field category: {category}")

    return {
        "label": spec["label"],
        "source": spec["source"],
        "category": category,
        "value_ours": value_ours,
        "value_bbg": value_bbg,
        "deviation_pct": dev_pct,
        "deviation_pp": dev_pp,
        "flag": flag,
        "screening_use": spec["screening_use"],
        "caveat": caveat,
    }


def validate_all(ticker: str, bbg_values: dict, api_key: str) -> dict:
    """Fetch every field for `ticker` and compare each against `bbg_values`."""
    roic = fetch_roic(ticker, api_key)
    cashflow = fetch_cashflow_fields(ticker)
    margin = fetch_margin_fields(ticker)
    balance_sheet = fetch_balance_sheet_fields(ticker, cashflow.get("ebitda_t12m"))
    valuation = fetch_valuation_fields(ticker)
    price = fetch_price_fields(ticker)
    shares = fetch_share_count_fields(ticker)

    ours = {
        "roic_current": roic.get("roic_current"),
        "roic_5y_avg": roic.get("roic_5y_avg"),
        "cfo_t12m": cashflow.get("cfo_t12m"),
        "net_income_t12m": cashflow.get("net_income_t12m"),
        "ebitda_t12m": cashflow.get("ebitda_t12m"),
        "ebitda_margin_3y_avg": margin.get("ebitda_margin_3y_avg"),
        "net_debt": balance_sheet.get("net_debt"),
        "net_debt_ebitda_ratio": balance_sheet.get("net_debt_ebitda_ratio"),
        "trailing_pe": valuation.get("trailing_pe"),
        "forward_pe": valuation.get("forward_pe"),
        "return_1y": price.get("return_1y"),
        "rsi_14d": price.get("rsi_14d"),
        "rsi_30d": price.get("rsi_30d"),
        "share_count_direction": shares.get("share_count_direction"),
        "share_count_cagr": shares.get("share_count_cagr"),
    }

    fields = {
        spec["key"]: _evaluate_field(spec, ours.get(spec["ours_key"]), bbg_values.get(spec["key"]))
        for spec in FIELD_SPECS
    }

    return {
        "ticker": ticker,
        "fields": fields,
        "raw": {
            "roic": roic, "cashflow": cashflow, "margin": margin,
            "balance_sheet": balance_sheet, "valuation": valuation,
            "price": price, "shares": shares,
        },
    }


# ============================================================
# Formatting / console + CSV output
# ============================================================

_MONEY_KEYS = {"cfo_t12m", "net_income_t12m", "ebitda_t12m", "net_debt"}
_PCT_KEYS = {"roic_current", "roic_5y_avg", "ebitda_margin_3y_avg", "return_1y", "share_count_5y_geo"}
_RATIO_KEYS = {"net_debt_ebitda"}
_PE_KEYS = {"trailing_pe", "forward_pe"}
_PLAIN_KEYS = {"rsi_14d", "rsi_30d"}  # RSI is an oscillator (0-100), not a percentage


def _fmt_value(key: str, value) -> str:
    if value is None:
        return "N/A"
    if key in _MONEY_KEYS:
        return f"{value:,.0f}M"
    if key in _PCT_KEYS:
        return f"{value:.2f}%"
    if key in _RATIO_KEYS:
        return f"{value:.3f}x"
    if key in _PE_KEYS:
        return f"{value:.2f}x"
    if key in _PLAIN_KEYS:
        return f"{value:.2f}"
    return str(value)


def _fmt_dev(field: dict, include_pp_note: bool = True) -> str:
    category = field["category"]
    if category in ("roic", "margin", "share_count_cagr"):
        return "N/A" if field["deviation_pp"] is None else f"{field['deviation_pp']:.2f}pp"
    if category == "return_rsi":
        if field["deviation_pct"] is None:
            return "N/A"
        pp_note = f" ({field['deviation_pp']:.2f}pp)" if include_pp_note and field["deviation_pp"] is not None else ""
        return f"{field['deviation_pct']:.2f}%{pp_note}"
    if category in ("monetary", "pe", "net_debt_ebitda", "net_debt_absolute"):
        return "N/A" if field["deviation_pct"] is None else f"{field['deviation_pct']:.2f}%"
    return "N/A"


def _print_ticker_report(result: dict) -> None:
    ticker = result["ticker"]
    print("=" * 40)
    print(f"BLOOMBERG FIELD VALIDATION — {ticker}")
    print("=" * 40)

    for spec in FIELD_SPECS:
        key = spec["key"]
        field = result["fields"][key]
        tag = "[FMP]" if field["source"] == "FMP" else "[YF] "
        print(f"\n{tag} {field['label']}")
        print(
            f"  Ours:  {_fmt_value(key, field['value_ours']):<10} "
            f"BBG: {_fmt_value(key, field['value_bbg']):<10} "
            f"Dev: {_fmt_dev(field):<12} [{_colorize(field['flag'])}]"
        )
        print(f"  Use:   {field['screening_use']}")
        if field["caveat"]:
            print(f"  Note:  {field['caveat']}")

    roic_error = result["raw"]["roic"].get("roic_error")
    if roic_error:
        print(f"\n  [FMP] ROIC fetch note: {roic_error}")
    print()


def _worse_flag(flag_a: str, flag_b: str) -> str:
    """For graded (GREEN/YELLOW/RED/N/A) categories, the higher-severity flag.

    DIRECTIONAL / APPROVED_WITH_CAVEAT categories always produce the same
    flag for both tickers (the category, not the data, drives the flag), so
    this only ever discriminates on the graded categories.
    """
    if flag_a == flag_b:
        return flag_a
    rank = {"GREEN": 0, "YELLOW": 1, "RED": 2, "N/A": 2}
    return flag_a if rank.get(flag_a, 0) >= rank.get(flag_b, 0) else flag_b


def _print_summary_table(results: dict) -> None:
    tickers = list(results.keys())
    label_w, dev_w, status_w = 24, 14, 23
    total_w = label_w + dev_w * len(tickers) + status_w + 6

    print("=" * total_w)
    print("FIELD VALIDATION SUMMARY")
    print("=" * total_w)
    header = f"{'Field':<{label_w}}" + "".join(f"{t:<{dev_w}}" for t in tickers) + f"{'Status':<{status_w}}Source"
    print(header)
    print("-" * total_w)

    for spec in FIELD_SPECS:
        key = spec["key"]
        fields = [results[t]["fields"][key] for t in tickers]
        devs = [_fmt_dev(f, include_pp_note=False) for f in fields]
        status = fields[0]["flag"]
        for f in fields[1:]:
            status = _worse_flag(status, f["flag"])
        row = f"{spec['label']:<{label_w}}" + "".join(f"{d:<{dev_w}}" for d in devs) + f"{status:<{status_w}}{spec['source']}"
        print(row)

    print("=" * total_w)


def _print_data_source_decision() -> None:
    n_fmp = sum(1 for s in FIELD_SPECS if s["source"] == "FMP")
    n_yf = sum(1 for s in FIELD_SPECS if s["source"] == "YF")
    n_total = len(FIELD_SPECS)

    print("DATA SOURCE DECISION:")
    print(f"  Primary: yfinance (free, reliable for {n_yf} of {n_total} fields)")
    print(f"  Secondary: FMP free tier (ROIC only, {n_fmp} field(s) -- covers large-caps, FMP paid $20/mo for full Russell 1000)")
    print("  Not replicable: Net Debt absolute value (financial subsidiary adjustments)")
    print("  Directional only: Trailing/Forward P/E, Net Debt/EBITDA tier, Share Count CAGR/direction")
    print("=" * 60)


def _write_csv(results: dict) -> None:
    CSV_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "ticker", "field", "label", "source", "value_ours", "value_bbg",
        "deviation_pct", "deviation_pp", "flag", "screening_use", "caveat",
    ]
    with open(CSV_OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ticker, result in results.items():
            for spec in FIELD_SPECS:
                field = result["fields"][spec["key"]]
                writer.writerow({
                    "ticker": ticker,
                    "field": spec["key"],
                    "label": field["label"],
                    "source": field["source"],
                    "value_ours": field["value_ours"],
                    "value_bbg": field["value_bbg"],
                    "deviation_pct": field["deviation_pct"],
                    "deviation_pp": field["deviation_pp"],
                    "flag": field["flag"],
                    "screening_use": field["screening_use"],
                    "caveat": field["caveat"],
                })
    logger.info("Wrote %s", CSV_OUTPUT_PATH)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    load_dotenv()
    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        raise RuntimeError("FMP_API_KEY not set — add it to .env")

    results = {}
    for ticker, bbg_values in TICKERS_BBG.items():
        logger.info("Validating %s...", ticker)
        results[ticker] = validate_all(ticker, bbg_values, api_key)

    for ticker in TICKERS_BBG:
        _print_ticker_report(results[ticker])

    _print_summary_table(results)
    _print_data_source_decision()
    _write_csv(results)


if __name__ == "__main__":
    main()
