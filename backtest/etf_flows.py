"""Fetches and caches BTC spot ETF daily flow data.

Standalone module: no dependency on backtest/data_fetcher.py. Two sources are
combined: a manually-downloaded SoSoValue CSV seeds the historical series
(aggregate flow only, most accurate for the past), and yfinance shares-
outstanding data extends it for any date more recent than the seed's last
date (per-ETF flow, implied from day-over-day shares changes x close price).
Where both sources cover the same date, the SoSoValue value wins.
"""

import logging
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SOSOVALUE_SEED_FILE = DATA_DIR / "btc_etf_flows_sosovalue.csv"
CACHE_FILE = DATA_DIR / "btc_etf_flows.csv"
CACHE_TTL_SECONDS = 24 * 60 * 60

ETF_TICKERS = ["IBIT", "FBTC", "ARKB", "BITB", "GBTC"]
ETF_FLOW_COLUMNS = [f"{ticker.lower()}_flow_musd" for ticker in ETF_TICKERS]
SOURCE_COLUMNS = ["daily_flow_musd", *ETF_FLOW_COLUMNS]

FLOW_COLUMNS = [
    "daily_flow_musd",
    *ETF_FLOW_COLUMNS,
    "ex_gbtc_flow_musd",
    "gbtc_outflow_pressure",
    "flow_5d_sum",
    "flow_20d_sum",
    "flow_sma200",
    "flow_vs_sma200",
    "flow_long_trend",
    "flow_score",
    "ex_gbtc_flow_score",
    "flow_regime",
    "flow_consecutive_negative",
    "flow_exit_accelerator",
]

ACCUMULATION_THRESHOLD_MUSD = 3000
DISTRIBUTION_THRESHOLD_MUSD = -3000
EXIT_ACCELERATOR_MIN_STREAK = 5
EXIT_ACCELERATOR_FLOW_5D_MUSD = -1000
GBTC_OUTFLOW_PRESSURE_MUSD = -100
GBTC_OUTFLOW_PRESSURE_MIN_STREAK = 3
MERGE_FFILL_LIMIT_DAYS = 3


def _is_cache_fresh(path: Path) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < CACHE_TTL_SECONDS


def _normalize_to_utc_dates(index: pd.Index) -> pd.DatetimeIndex:
    """Collapse a (possibly tz-aware) DatetimeIndex to midnight-UTC dates."""
    return pd.to_datetime(pd.Index(index).date).tz_localize("UTC")


def _load_merged_cache(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.index.name = "date"
    return df


def _save_merged_cache(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path)


def _load_sosovalue_seed(path) -> pd.DataFrame:
    """Parse the SoSoValue seed CSV into a daily_flow_musd frame, sorted ascending."""
    raw = pd.read_csv(path, sep="\t")
    raw = raw.dropna(subset=["Date"])

    dates = pd.to_datetime(raw["Date"], format="%m/%d/%Y", utc=True)
    daily_flow_usd = (
        raw["Daily Total BTC Inflow(USD)"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .astype(float)
    )

    seed = pd.DataFrame({"daily_flow_musd": daily_flow_usd.values / 1_000_000}, index=dates)
    seed.index.name = "date"
    seed = seed.sort_index()
    seed = seed[~seed.index.duplicated(keep="last")]
    return seed


def _fetch_single_etf_flow(ticker: str, start_date: pd.Timestamp) -> pd.Series:
    """Implied daily flow for one ETF: shares_outstanding.diff() x closing price, in $musd."""
    empty = pd.Series(
        dtype=float, index=pd.DatetimeIndex([], tz="UTC"), name=f"{ticker.lower()}_flow_musd"
    )

    t = yf.Ticker(ticker)
    shares = t.get_shares_full(start=start_date.date())
    if shares is None or len(shares) == 0:
        return empty

    shares.index = _normalize_to_utc_dates(shares.index)
    shares = shares[~shares.index.duplicated(keep="last")].sort_index()
    shares_change = shares.diff()

    hist = t.history(start=start_date.date())
    if hist.empty:
        return empty

    close = hist["Close"]
    close.index = _normalize_to_utc_dates(close.index)
    close = close[~close.index.duplicated(keep="last")].sort_index()

    implied_flow_usd = (shares_change * close).dropna()
    return (implied_flow_usd / 1_000_000).rename(f"{ticker.lower()}_flow_musd")


def _fetch_yfinance_extension(start_date: pd.Timestamp) -> pd.DataFrame:
    """Per-ETF implied flow from yfinance shares outstanding, plus the aggregate total."""
    per_etf = {}
    for ticker in ETF_TICKERS:
        col = f"{ticker.lower()}_flow_musd"
        try:
            per_etf[col] = _fetch_single_etf_flow(ticker, start_date)
        except Exception as exc:
            logger.warning("yfinance flow fetch failed for %s (%s)", ticker, exc)
            per_etf[col] = pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"), name=col)

    combined = pd.concat(per_etf.values(), axis=1, sort=True)
    combined.index.name = "date"
    combined["daily_flow_musd"] = combined[ETF_FLOW_COLUMNS].sum(axis=1, min_count=1)
    return combined


def _merge_seed_and_extension(seed: pd.DataFrame, extension: pd.DataFrame) -> pd.DataFrame:
    """SoSoValue as base; yfinance extends past the seed's last date. SoSoValue wins on overlap."""
    all_dates = seed.index.union(extension.index)
    seed_aligned = seed.reindex(all_dates)
    ext_aligned = extension.reindex(all_dates, columns=SOURCE_COLUMNS)

    combined = ext_aligned.copy()
    combined["daily_flow_musd"] = seed_aligned["daily_flow_musd"].combine_first(
        ext_aligned["daily_flow_musd"]
    )
    combined = combined.sort_index()

    full_calendar = pd.date_range(combined.index.min(), combined.index.max(), freq="D", tz="UTC")
    combined = combined.reindex(full_calendar).ffill(limit=MERGE_FFILL_LIMIT_DAYS)
    combined.index.name = "date"
    return combined


def _consecutive_streak(condition: pd.Series) -> pd.Series:
    """Length of the current run of consecutive True values ending at each row (0 elsewhere).

    NaN/missing input is treated as False, not as a streak-breaking special case, so the
    run-length groupby stays correct across gaps (see tests/test_indicators.py).
    """
    cond = condition.fillna(False)
    run_id = (cond != cond.shift()).cumsum()
    streak = cond.groupby(run_id).cumcount() + 1
    return streak.where(cond, 0)


def _bucket_flow_score(rolling_5d_sum: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [
                rolling_5d_sum > 500,
                rolling_5d_sum > 100,
                rolling_5d_sum >= -100,
                rolling_5d_sum >= -500,
            ],
            [1.0, 0.75, 0.5, 0.25],
            default=0.0,
        ),
        index=rolling_5d_sum.index,
    ).mask(rolling_5d_sum.isna())


def _compute_derived_metrics(merged: pd.DataFrame) -> pd.DataFrame:
    daily_flow_musd = merged["daily_flow_musd"]

    flow_5d_sum = daily_flow_musd.rolling(5).sum().rename("flow_5d_sum")
    flow_20d_sum = daily_flow_musd.rolling(20).sum().rename("flow_20d_sum")
    flow_sma200 = daily_flow_musd.rolling(200).mean().rename("flow_sma200")
    flow_vs_sma200 = (daily_flow_musd - flow_sma200).rename("flow_vs_sma200")

    flow_long_trend = pd.Series(
        np.where(flow_20d_sum > flow_sma200 * 20, 1, -1),
        index=daily_flow_musd.index,
        dtype=float,
    ).mask(flow_sma200.isna())
    flow_long_trend.name = "flow_long_trend"

    flow_score = _bucket_flow_score(flow_5d_sum).rename("flow_score")

    ibit = merged["ibit_flow_musd"].rename("ibit_flow_musd")
    fbtc = merged["fbtc_flow_musd"].rename("fbtc_flow_musd")
    arkb = merged["arkb_flow_musd"].rename("arkb_flow_musd")
    bitb = merged["bitb_flow_musd"].rename("bitb_flow_musd")
    gbtc = merged["gbtc_flow_musd"].rename("gbtc_flow_musd")

    ex_gbtc_flow_musd = pd.concat([ibit, fbtc, arkb, bitb], axis=1).sum(axis=1, min_count=1)
    ex_gbtc_flow_musd.name = "ex_gbtc_flow_musd"

    # Bucketed on its own 5d sum, same convention as flow_score, so the two are
    # directly comparable; falls back to flow_score wherever per-ETF data is absent
    # (i.e. the SoSoValue-only history, which has no ex-GBTC breakdown).
    ex_gbtc_5d_sum = ex_gbtc_flow_musd.rolling(5).sum()
    ex_gbtc_flow_score = (
        _bucket_flow_score(ex_gbtc_5d_sum).fillna(flow_score).rename("ex_gbtc_flow_score")
    )

    gbtc_pressure_streak = _consecutive_streak(gbtc < GBTC_OUTFLOW_PRESSURE_MUSD)
    gbtc_outflow_pressure = (gbtc_pressure_streak >= GBTC_OUTFLOW_PRESSURE_MIN_STREAK).rename(
        "gbtc_outflow_pressure"
    )

    flow_regime = pd.Series(
        np.select(
            [
                flow_20d_sum > ACCUMULATION_THRESHOLD_MUSD,
                flow_20d_sum < DISTRIBUTION_THRESHOLD_MUSD,
            ],
            ["accumulation", "distribution"],
            default="neutral",
        ),
        index=daily_flow_musd.index,
    ).mask(flow_20d_sum.isna())
    flow_regime.name = "flow_regime"

    flow_consecutive_negative = _consecutive_streak(daily_flow_musd < 0).rename(
        "flow_consecutive_negative"
    )

    flow_exit_accelerator = (
        (flow_consecutive_negative >= EXIT_ACCELERATOR_MIN_STREAK)
        & (flow_5d_sum < EXIT_ACCELERATOR_FLOW_5D_MUSD)
    ).rename("flow_exit_accelerator")

    return pd.concat(
        [
            daily_flow_musd,
            ibit, fbtc, arkb, bitb, gbtc,
            ex_gbtc_flow_musd, gbtc_outflow_pressure,
            flow_5d_sum, flow_20d_sum, flow_sma200, flow_vs_sma200, flow_long_trend,
            flow_score, ex_gbtc_flow_score, flow_regime,
            flow_consecutive_negative, flow_exit_accelerator,
        ],
        axis=1,
    )


def _empty_flow_frame() -> pd.DataFrame:
    df = pd.DataFrame(columns=FLOW_COLUMNS)
    df.index.name = "date"
    return df


def fetch_etf_flows(days_back: int = 1460) -> pd.DataFrame:
    """Fetch BTC spot ETF daily net flows with derived metrics.

    SoSoValue CSV seeds the historical aggregate; yfinance shares-outstanding
    extends it past the seed's last date with a per-ETF breakdown. Derived
    metrics (rolling sums, SMA200, scores, regime, streaks) are computed on
    the full merged history so early rows of the returned window aren't
    warm-up NaNs, then the frame is trimmed to the trailing `days_back` days.

    On any fetch failure: logs a warning and falls back to the cached merged
    data if available; if no cache exists either, returns an empty DataFrame
    with the correct columns rather than raising.
    """
    try:
        if _is_cache_fresh(CACHE_FILE):
            merged = _load_merged_cache(CACHE_FILE)
        else:
            seed = _load_sosovalue_seed(SOSOVALUE_SEED_FILE)
            extension_start = seed.index.max() + pd.Timedelta(days=1)
            extension = _fetch_yfinance_extension(extension_start)
            merged = _merge_seed_and_extension(seed, extension)
            _save_merged_cache(merged, CACHE_FILE)
        df = _compute_derived_metrics(merged)
    except Exception as exc:
        logger.warning("ETF flow fetch failed (%s); falling back to cache", exc)
        if not CACHE_FILE.exists():
            return _empty_flow_frame()
        merged = _load_merged_cache(CACHE_FILE)
        df = _compute_derived_metrics(merged)

    if days_back is not None and not df.empty:
        cutoff = df.index.max() - timedelta(days=days_back)
        df = df[df.index > cutoff]

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = fetch_etf_flows()

    print(f"shape: {result.shape}")
    if not result.empty:
        print(f"date range: {result.index.min()} to {result.index.max()}")

    print("\nlast 10 rows:")
    display_cols = ["daily_flow_musd", "flow_5d_sum", "flow_score", "flow_regime", "flow_long_trend"]
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(result[display_cols].tail(10))

    if not result.empty:
        latest = result.iloc[-1]
        print(f"\ncurrent flow_regime: {latest['flow_regime']}")
        print(f"current flow_score: {latest['flow_score']}")

        sosovalue_days = result["ibit_flow_musd"].isna().sum()
        yfinance_days = result["ibit_flow_musd"].notna().sum()
        print(f"\ndays from SoSoValue (aggregate only): {sosovalue_days}")
        print(f"days from yfinance extension (per-ETF): {yfinance_days}")
