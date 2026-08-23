"""Validates FMP data quality against Bloomberg Terminal values.

Pulls profitability, balance-sheet, growth, and valuation metrics from FMP
plus price/technical data from yfinance for a fixed watchlist, then writes a
CSV template with the corresponding Bloomberg fields left blank for manual
entry from the Terminal. Deviation columns are computed automatically —
they read as N/A until a Bloomberg value has been filled in.

Re-running the script preserves any Bloomberg values already saved in
`equity/data/bloomberg_validation_template.csv` (matched by ticker) while
refreshing the FMP/yfinance columns and recomputing deviations, so filling
in the Terminal and re-running is a safe iterative workflow.

Usage: python -m equity.tools.bloomberg_validation

Note on FMP endpoints: FMP retired its legacy `/api/v3/*` endpoints on
August 31, 2025 (403 for any key without a pre-existing legacy subscription),
so this uses the current `/stable/*` API instead. Two fields moved endpoints
in that migration — `netProfitMargin` and `operatingCashFlowPerShare` now
live under `/stable/ratios` rather than `/stable/key-metrics` — and
`debtToEquity` no longer exists by that name; the closest equivalent is
`debtToEquityRatio` in `/stable/ratios`. `/stable` also has no plain
forward-P/E field, so `forward_pe_fmp` is derived as current price
(`/stable/quote`) divided by the next fiscal year's consensus EPS estimate
(`/stable/analyst-estimates`).
"""

import logging
import os
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

from backtest.indicators import rsi

logger = logging.getLogger(__name__)

TICKERS = ["APP", "MSFT", "CBRE", "ADP", "CCJ", "CEG", "ADBE", "GDDY", "PGR", "WRB", "UMAC"]

FMP_BASE_URL = "https://financialmodelingprep.com/stable"
REQUEST_TIMEOUT_SECONDS = 15

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUTPUT_CSV = DATA_DIR / "bloomberg_validation_template.csv"

# Column order defines the CSV layout: each FMP field sits next to its
# Bloomberg counterpart (and deviation, where requested) for easy review.
COLUMN_ORDER = [
    "ticker",
    "roic_fmp", "roic_bbg", "roic_deviation_pct",
    "net_margin_fmp", "net_margin_bbg",
    "cfo_per_share_fmp",
    "net_debt_ebitda_fmp", "net_debt_ebitda_bbg", "net_debt_ebitda_deviation_pct",
    "debt_equity_fmp",
    "rev_growth_1y_fmp", "rev_growth_1y_bbg",
    "eps_growth_1y_fmp",
    "share_count_growth_fmp", "share_count_growth_bbg",
    "trailing_pe_fmp", "trailing_pe_bbg", "pe_deviation_pct",
    "forward_pe_fmp", "forward_pe_bbg",
    "ebitda_margin_fmp", "ebitda_margin_bbg", "ebitda_margin_deviation_pct",
    "return_1y_yf", "return_1y_bbg",
    "return_5y_yf",
    "rsi_14d_yf", "rsi_14d_bbg",
    "rsi_30d_yf",
]

BBG_FIELDS = [c for c in COLUMN_ORDER if c.endswith("_bbg")]
DEVIATION_FIELDS = [c for c in COLUMN_ORDER if c.endswith("_deviation_pct")]
FMP_YF_FIELDS = [c for c in COLUMN_ORDER if c not in ("ticker", *BBG_FIELDS, *DEVIATION_FIELDS)]

# deviation field -> (fmp field, bbg field) it's computed from.
DEVIATION_SPECS = {
    "roic_deviation_pct": ("roic_fmp", "roic_bbg"),
    "net_debt_ebitda_deviation_pct": ("net_debt_ebitda_fmp", "net_debt_ebitda_bbg"),
    "pe_deviation_pct": ("trailing_pe_fmp", "trailing_pe_bbg"),
    "ebitda_margin_deviation_pct": ("ebitda_margin_fmp", "ebitda_margin_bbg"),
}


def _fmp_get(endpoint: str, ticker: str, api_key: str, **extra_params) -> dict:
    """Fetch the single most recent record for `ticker` from an FMP `/stable` endpoint.

    Returns an empty dict (and logs a warning) on any request failure or
    on an empty/unexpected response, e.g. a ticker FMP doesn't cover.
    """
    url = f"{FMP_BASE_URL}/{endpoint}"
    params = {"symbol": ticker, "limit": 1, "apikey": api_key, **extra_params}
    try:
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        logger.warning("FMP %s request failed for %s: %s", endpoint, ticker, exc)
        return {}

    if not isinstance(data, list) or not data:
        logger.warning("FMP %s returned no data for %s (not covered?)", endpoint, ticker)
        return {}
    return data[0]


def _fetch_forward_pe(ticker: str, api_key: str, last_reported_date: str | None):
    """Derive forward P/E as current price / next fiscal year's consensus EPS.

    FMP's `/stable` plan doesn't expose a plain forwardPE field, so this
    combines `/stable/quote` (current price) with `/stable/analyst-estimates`
    (consensus EPS for the nearest fiscal year not yet reported). Returns
    "N/A" if either leg is unavailable.
    """
    quote = _fmp_get("quote", ticker, api_key)
    price = quote.get("price")
    if price is None:
        logger.warning("FMP quote returned no price for %s — forward_pe_fmp N/A", ticker)
        return "N/A"

    url = f"{FMP_BASE_URL}/analyst-estimates"
    params = {"symbol": ticker, "period": "annual", "limit": 10, "apikey": api_key}
    try:
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        resp.raise_for_status()
        estimates = resp.json()
    except requests.RequestException as exc:
        logger.warning("FMP analyst-estimates request failed for %s: %s", ticker, exc)
        return "N/A"

    if not isinstance(estimates, list) or not estimates:
        logger.warning("FMP analyst-estimates returned no data for %s", ticker)
        return "N/A"

    future = [
        e
        for e in estimates
        if e.get("date") and e.get("epsAvg") is not None
        and (last_reported_date is None or e["date"] > last_reported_date)
    ]
    if not future:
        logger.warning("No forward-looking EPS estimate found for %s — forward_pe_fmp N/A", ticker)
        return "N/A"

    forward_eps = min(future, key=lambda e: e["date"])["epsAvg"]
    if not forward_eps:
        return "N/A"
    return price / forward_eps


def _fetch_fmp_fields(ticker: str, api_key: str) -> dict:
    """Fetch profitability, balance-sheet, growth, and valuation fields from FMP."""
    key_metrics = _fmp_get("key-metrics", ticker, api_key)
    growth = _fmp_get("financial-growth", ticker, api_key)
    ratios = _fmp_get("ratios", ticker, api_key)

    raw = {
        "roic_fmp": key_metrics.get("returnOnCapitalEmployed"),
        "net_margin_fmp": ratios.get("netProfitMargin"),
        "cfo_per_share_fmp": ratios.get("operatingCashFlowPerShare"),
        "net_debt_ebitda_fmp": key_metrics.get("netDebtToEBITDA"),
        "debt_equity_fmp": ratios.get("debtToEquityRatio"),
        "rev_growth_1y_fmp": growth.get("revenueGrowth"),
        "eps_growth_1y_fmp": growth.get("epsgrowth"),
        "share_count_growth_fmp": growth.get("weightedAverageSharesGrowth"),
        "trailing_pe_fmp": ratios.get("priceEarningsRatio"),
        "ebitda_margin_fmp": ratios.get("ebitdaMargin"),
    }
    raw["forward_pe_fmp"] = _fetch_forward_pe(ticker, api_key, key_metrics.get("date"))
    return {k: ("N/A" if v is None else v) for k, v in raw.items()}


def _total_return(closes: pd.Series, years: float):
    """Total return from ~`years` ago to the latest close.

    Returns "N/A" if price history doesn't reach back far enough. A small
    tolerance absorbs the case where yfinance's `period=` window is anchored
    on today's calendar date rather than the last trading date, so the
    earliest bar can land a day or two after the exact target date even when
    the full window of history is actually present.
    """
    last_date = closes.index[-1]
    target_date = last_date - pd.DateOffset(years=years)
    if target_date < closes.index[0] - pd.Timedelta(days=10):
        return "N/A"

    idx = min(max(closes.index.searchsorted(target_date), 0), len(closes) - 1)
    base_price = closes.iloc[idx]
    if pd.isna(base_price) or base_price == 0:
        return "N/A"
    return closes.iloc[-1] / base_price - 1


def _fetch_yf_fields(ticker: str) -> dict:
    """Fetch 1Y/5Y total return and 14D/30D RSI from yfinance daily closes."""
    fields = ["return_1y_yf", "return_5y_yf", "rsi_14d_yf", "rsi_30d_yf"]
    try:
        hist = yf.Ticker(ticker).history(period="5y", auto_adjust=True)
        closes = hist["Close"].dropna()
        if closes.empty:
            raise ValueError("no price history returned")
    except Exception as exc:
        logger.warning("yfinance fetch failed for %s: %s", ticker, exc)
        return {f: "N/A" for f in fields}

    rsi_14 = rsi(closes, 14).iloc[-1]
    rsi_30 = rsi(closes, 30).iloc[-1]

    return {
        "return_1y_yf": _total_return(closes, years=1),
        "return_5y_yf": _total_return(closes, years=5),
        "rsi_14d_yf": "N/A" if pd.isna(rsi_14) else rsi_14,
        "rsi_30d_yf": "N/A" if pd.isna(rsi_30) else rsi_30,
    }


def _deviation_pct(fmp_val, bbg_val):
    """abs(fmp - bbg) / abs(bbg) * 100, or "N/A" if either side is missing."""
    if fmp_val in (None, "", "N/A") or bbg_val in (None, "", "N/A"):
        return "N/A"
    try:
        fmp_f, bbg_f = float(fmp_val), float(bbg_val)
    except (TypeError, ValueError):
        return "N/A"
    if bbg_f == 0:
        return "N/A"
    return abs(fmp_f - bbg_f) / abs(bbg_f) * 100


def _load_existing_bbg_values(path: Path) -> dict:
    """Read Bloomberg columns already filled in on disk, keyed by ticker."""
    if not path.exists():
        return {}
    try:
        existing = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as exc:
        logger.warning("Could not read existing template at %s: %s", path, exc)
        return {}
    if "ticker" not in existing.columns:
        return {}
    return {
        row["ticker"]: {field: row.get(field, "") for field in BBG_FIELDS}
        for _, row in existing.iterrows()
    }


def validate() -> pd.DataFrame:
    """Build the full validation table: one row per ticker in `TICKERS`."""
    load_dotenv()
    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        raise RuntimeError("FMP_API_KEY not set — add it to .env")

    existing_bbg = _load_existing_bbg_values(OUTPUT_CSV)

    rows = []
    for ticker in TICKERS:
        logger.info("Fetching %s...", ticker)
        row = {"ticker": ticker}
        row.update(_fetch_fmp_fields(ticker, api_key))
        row.update(_fetch_yf_fields(ticker))

        saved_bbg = existing_bbg.get(ticker, {})
        for field in BBG_FIELDS:
            row[field] = saved_bbg.get(field, "")

        for dev_field, (fmp_field, bbg_field) in DEVIATION_SPECS.items():
            row[dev_field] = _deviation_pct(row.get(fmp_field), row.get(bbg_field))

        rows.append(row)

    return pd.DataFrame(rows, columns=COLUMN_ORDER)


def _print_summary(df: pd.DataFrame) -> None:
    summary = df[["ticker", *FMP_YF_FIELDS]].copy()
    for col in FMP_YF_FIELDS:
        summary[col] = summary[col].apply(
            lambda v: round(v, 4) if isinstance(v, (int, float)) else v
        )
    print("\n=== FMP / yfinance Data Summary ===\n")
    print(summary.to_string(index=False))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    df = validate()
    _print_summary(df)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nValidation complete. Template written to {OUTPUT_CSV}")
    print("Fill in the *_bbg columns from Bloomberg Terminal, then re-run to compute deviations.")


if __name__ == "__main__":
    main()
