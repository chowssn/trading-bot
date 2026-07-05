"""Fetches and caches historical market/macro data for backtesting.

Coinbase candles, yfinance macro series, and FRED credit spreads are each
cached as CSVs in backtest/data/ and reused for 24 hours before refetching.
"""

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv
from fredapi import Fred

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CACHE_TTL_SECONDS = 24 * 60 * 60

COINBASE_CANDLES_URL = (
    "https://api.coinbase.com/api/v3/brokerage/market/products/BTC-USDC/candles"
)
COINBASE_MAX_CANDLES_PER_REQUEST = 350


def _is_cache_fresh(path: Path) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < CACHE_TTL_SECONDS


def _load_cached_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


def _save_cache(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path)


def _normalize_to_utc_dates(index: pd.Index) -> pd.DatetimeIndex:
    """Collapse a (possibly tz-aware) DatetimeIndex to midnight-UTC dates."""
    return pd.to_datetime(pd.Index(index).date).tz_localize("UTC")


def fetch_btc_candles(days_back: int = 1460) -> pd.DataFrame:
    """Fetch BTC-USDC daily OHLCV from Coinbase's public candles endpoint."""
    cache_file = DATA_DIR / f"btc_candles_{days_back}d.csv"
    if _is_cache_fresh(cache_file):
        return _load_cached_csv(cache_file)

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days_back)
    chunk_delta = timedelta(days=COINBASE_MAX_CANDLES_PER_REQUEST - 1)

    all_candles = []
    chunk_end = end_time
    while chunk_end > start_time:
        chunk_start = max(start_time, chunk_end - chunk_delta)
        params = {
            "granularity": "ONE_DAY",
            "start": str(int(chunk_start.timestamp())),
            "end": str(int(chunk_end.timestamp())),
        }
        response = requests.get(COINBASE_CANDLES_URL, params=params, timeout=15)
        response.raise_for_status()
        all_candles.extend(response.json().get("candles", []))
        chunk_end = chunk_start
        time.sleep(0.2)

    if not all_candles:
        raise ValueError("No candle data returned from Coinbase")

    df = pd.DataFrame(all_candles)
    df["timestamp"] = pd.to_datetime(df["start"].astype(int), unit="s", utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)

    df = (
        df[["timestamp", "open", "high", "low", "close", "volume"]]
        .sort_values("timestamp")
        .drop_duplicates(subset="timestamp")
        .set_index("timestamp")
    )

    _save_cache(df, cache_file)
    return df


def fetch_macro_data(days_back: int = 1460) -> pd.DataFrame:
    """Fetch QQQ, UUP, and VIX daily closes via yfinance, ffilled to a 7-day calendar."""
    cache_file = DATA_DIR / f"macro_data_{days_back}d.csv"
    if _is_cache_fresh(cache_file):
        return _load_cached_csv(cache_file)

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days_back)

    tickers = {"QQQ": "qqq_close", "UUP": "uup_close", "^VIX": "vix_close"}
    series_list = []
    for ticker, col_name in tickers.items():
        hist = yf.Ticker(ticker).history(
            start=start_date.date(), end=end_date.date() + timedelta(days=1)
        )
        close = hist["Close"].rename(col_name)
        close.index = _normalize_to_utc_dates(close.index)
        series_list.append(close)

    df = pd.concat(series_list, axis=1, sort=True).sort_index()

    full_index = pd.date_range(
        start=start_date.replace(hour=0, minute=0, second=0, microsecond=0),
        end=end_date.replace(hour=0, minute=0, second=0, microsecond=0),
        freq="D",
        tz="UTC",
    )
    df = df.reindex(full_index).ffill()
    df.index.name = "date"

    _save_cache(df, cache_file)
    return df


def fetch_credit_spreads(days_back: int = 1460) -> pd.DataFrame:
    """Fetch ICE BofA US High Yield OAS (BAMLH0A0HYM2) from FRED, ffilled daily."""
    cache_file = DATA_DIR / f"credit_spreads_{days_back}d.csv"
    if _is_cache_fresh(cache_file):
        return _load_cached_csv(cache_file)

    load_dotenv()
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("FRED_API_KEY not set in .env")

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days_back)

    fred = Fred(api_key=api_key)
    series = fred.get_series(
        "BAMLH0A0HYM2",
        observation_start=start_date.date(),
        observation_end=end_date.date(),
    )
    series.index = _normalize_to_utc_dates(series.index)
    series.name = "hy_spread"

    full_index = pd.date_range(
        start=start_date.replace(hour=0, minute=0, second=0, microsecond=0),
        end=end_date.replace(hour=0, minute=0, second=0, microsecond=0),
        freq="D",
        tz="UTC",
    )
    df = series.to_frame().reindex(full_index).ffill()
    df.index.name = "date"

    _save_cache(df, cache_file)
    return df


def fetch_all(days_back: int = 1460) -> pd.DataFrame:
    """Fetch BTC candles, macro data, and credit spreads and merge into one frame."""
    btc = fetch_btc_candles(days_back)
    macro = fetch_macro_data(days_back)
    credit = fetch_credit_spreads(days_back)

    combined = btc.join(macro, how="outer").join(credit, how="outer")
    combined = combined.sort_index().ffill()
    combined = combined.dropna(subset=["close"])
    return combined


if __name__ == "__main__":
    result = fetch_all()
    print(f"shape: {result.shape}")
    print(f"date range: {result.index.min()} to {result.index.max()}")
    print(result.head(3))
