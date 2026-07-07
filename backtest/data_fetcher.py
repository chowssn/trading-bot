"""Fetches and caches historical instrument/macro data for backtesting.

Coinbase candles, yfinance macro series, and FRED credit spreads are each
cached as CSVs in backtest/data/ and reused for 24 hours before refetching.
"""

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv
from fredapi import Fred

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CACHE_TTL_SECONDS = 24 * 60 * 60

COINBASE_CANDLES_URL_TEMPLATE = (
    "https://api.coinbase.com/api/v3/brokerage/market/products/{symbol}/candles"
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


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def _trend_binary(a: pd.Series, b: pd.Series, invert: bool = False) -> pd.Series:
    """+1 if a>b else -1 (or the reverse if `invert`); NaN while either input is NaN."""
    cond = a < b if invert else a > b
    trend = pd.Series(np.where(cond, 1, -1), index=a.index, dtype=float)
    return trend.mask(a.isna() | b.isna())


def _trend_banded(a: pd.Series, b: pd.Series, band: float = 0.005) -> pd.Series:
    """+1/-1 if a is more than `band` fraction above/below b, else 0; NaN while either input is NaN."""
    diff = (a - b) / b
    trend = pd.Series(
        np.where(diff > band, 1, np.where(diff < -band, -1, 0)), index=a.index, dtype=float
    )
    return trend.mask(a.isna() | b.isna())


def fetch_instrument_data(symbol: str = "BTC-USDC", days_back: int = 730) -> pd.DataFrame:
    """Fetch daily OHLCV for `symbol` from Coinbase's public candles endpoint."""
    cache_file = DATA_DIR / f"{symbol}_ohlcv.csv"
    if _is_cache_fresh(cache_file):
        return _load_cached_csv(cache_file)

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days_back)
    chunk_delta = timedelta(days=COINBASE_MAX_CANDLES_PER_REQUEST - 1)

    candles_url = COINBASE_CANDLES_URL_TEMPLATE.format(symbol=symbol)

    all_candles = []
    chunk_end = end_time
    while chunk_end > start_time:
        chunk_start = max(start_time, chunk_end - chunk_delta)
        params = {
            "granularity": "ONE_DAY",
            "start": str(int(chunk_start.timestamp())),
            "end": str(int(chunk_end.timestamp())),
        }
        response = requests.get(candles_url, params=params, timeout=15)
        response.raise_for_status()
        all_candles.extend(response.json().get("candles", []))
        chunk_end = chunk_start
        time.sleep(0.2)

    if not all_candles:
        raise ValueError(f"No candle data returned from Coinbase for {symbol}")

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


def fetch_macro_data(days_back: int = 730) -> pd.DataFrame:
    """Fetch SPY, QQQ, UUP, VIX, CPER (yfinance) and FRED HY credit spreads.

    Trend/moving-average columns are computed on each series' native trading-day
    frequency (so weekend repeats from forward-fill don't distort EMA/SMA), then
    the whole frame is forward-filled onto BTC's 7-day calendar.

    QQQ vs SPY relative strength (qqq_spy_ratio and its trend columns) tracks tech
    leadership over the broad market: qqq_spy_short_trend and qqq_spy_long_trend
    both positive means sustained tech leadership, read as a risk-on confirmation
    for BTC.

    UUP (DXY proxy) trend columns are inverted: a weakening dollar (uup below its
    moving average) is bullish, so uup_short_trend/uup_long_trend are +1 in that case.

    Copper (CPER) is a global-growth proxy: copper_long_trend captures multi-month
    growth cycles, copper_short_trend captures near-term momentum.
    """
    cache_file = DATA_DIR / "macro_v2.csv"
    if _is_cache_fresh(cache_file):
        return _load_cached_csv(cache_file)

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days_back)

    tickers = {
        "SPY": "spy_close",
        "QQQ": "qqq_close",
        "UUP": "uup_close",
        "^VIX": "vix_close",
        "CPER": "copper_close",
    }
    closes = {}
    for ticker, col_name in tickers.items():
        hist = yf.Ticker(ticker).history(
            start=start_date.date(), end=end_date.date() + timedelta(days=1)
        )
        close = hist["Close"].rename(col_name)
        close.index = _normalize_to_utc_dates(close.index)
        closes[col_name] = close

    prices = pd.concat(closes.values(), axis=1, sort=True).sort_index()

    spy_ema10 = _ema(prices["spy_close"], 10).rename("spy_ema10")
    spy_sma30 = _sma(prices["spy_close"], 30).rename("spy_sma30")
    spy_sma200 = _sma(prices["spy_close"], 200).rename("spy_sma200")
    spy_short_trend = _trend_banded(spy_ema10, spy_sma30).rename("spy_short_trend")
    spy_long_trend = _trend_binary(prices["spy_close"], spy_sma200).rename("spy_long_trend")

    qqq_ema10 = _ema(prices["qqq_close"], 10).rename("qqq_ema10")
    qqq_sma30 = _sma(prices["qqq_close"], 30).rename("qqq_sma30")
    qqq_sma200 = _sma(prices["qqq_close"], 200).rename("qqq_sma200")
    qqq_short_trend = _trend_banded(qqq_ema10, qqq_sma30).rename("qqq_short_trend")
    qqq_long_trend = _trend_binary(prices["qqq_close"], qqq_sma200).rename("qqq_long_trend")

    qqq_spy_ratio = (prices["qqq_close"] / prices["spy_close"]).rename("qqq_spy_ratio")
    qqq_spy_ratio_ema10 = _ema(qqq_spy_ratio, 10).rename("qqq_spy_ratio_ema10")
    qqq_spy_ratio_sma30 = _sma(qqq_spy_ratio, 30).rename("qqq_spy_ratio_sma30")
    qqq_spy_ratio_sma200 = _sma(qqq_spy_ratio, 200).rename("qqq_spy_ratio_sma200")
    qqq_spy_short_trend = _trend_binary(qqq_spy_ratio_ema10, qqq_spy_ratio_sma30).rename(
        "qqq_spy_short_trend"
    )
    qqq_spy_long_trend = _trend_binary(qqq_spy_ratio, qqq_spy_ratio_sma200).rename(
        "qqq_spy_long_trend"
    )

    uup_ema10 = _ema(prices["uup_close"], 10).rename("uup_ema10")
    uup_sma30 = _sma(prices["uup_close"], 30).rename("uup_sma30")
    uup_sma200 = _sma(prices["uup_close"], 200).rename("uup_sma200")
    uup_short_trend = _trend_binary(uup_ema10, uup_sma30, invert=True).rename("uup_short_trend")
    uup_long_trend = _trend_binary(prices["uup_close"], uup_sma200, invert=True).rename(
        "uup_long_trend"
    )

    copper_ema10 = _ema(prices["copper_close"], 10).rename("copper_ema10")
    copper_sma30 = _sma(prices["copper_close"], 30).rename("copper_sma30")
    copper_sma200 = _sma(prices["copper_close"], 200).rename("copper_sma200")
    copper_short_trend = _trend_binary(copper_ema10, copper_sma30).rename("copper_short_trend")
    copper_long_trend = _trend_binary(prices["copper_close"], copper_sma200).rename(
        "copper_long_trend"
    )

    load_dotenv()
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("FRED_API_KEY not set in .env")

    fred = Fred(api_key=api_key)
    hy_spread = fred.get_series(
        "BAMLH0A0HYM2",
        observation_start=start_date.date(),
        observation_end=end_date.date(),
    )
    hy_spread.index = _normalize_to_utc_dates(hy_spread.index)
    hy_spread.name = "hy_spread"

    df = pd.concat(
        [
            prices,
            spy_ema10, spy_sma30, spy_sma200, spy_short_trend, spy_long_trend,
            qqq_ema10, qqq_sma30, qqq_sma200, qqq_short_trend, qqq_long_trend,
            qqq_spy_ratio, qqq_spy_ratio_ema10, qqq_spy_ratio_sma30, qqq_spy_ratio_sma200,
            qqq_spy_short_trend, qqq_spy_long_trend,
            uup_ema10, uup_sma30, uup_sma200, uup_short_trend, uup_long_trend,
            copper_ema10, copper_sma30, copper_sma200, copper_short_trend, copper_long_trend,
            hy_spread,
        ],
        axis=1,
        sort=True,
    ).sort_index()

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


def fetch_all(symbol: str = "BTC-USDC", days_back: int = 730) -> pd.DataFrame:
    """Fetch instrument OHLCV and macro data and merge into one frame."""
    instrument = fetch_instrument_data(symbol, days_back)
    macro = fetch_macro_data(days_back)

    combined = instrument.join(macro, how="outer")
    combined = combined.sort_index().ffill()
    combined = combined.dropna(subset=["close"])
    return combined


if __name__ == "__main__":
    result = fetch_all("BTC-USDC")
    print(f"shape: {result.shape}")
    print(f"date range: {result.index.min()} to {result.index.max()}")
    print(f"columns: {list(result.columns)}")

    new_columns = [
        "spy_close", "spy_ema10", "spy_sma30", "spy_sma200", "spy_short_trend", "spy_long_trend",
        "qqq_ema10", "qqq_sma30", "qqq_sma200", "qqq_short_trend", "qqq_long_trend",
        "qqq_spy_ratio", "qqq_spy_ratio_ema10", "qqq_spy_ratio_sma30", "qqq_spy_ratio_sma200",
        "qqq_spy_short_trend", "qqq_spy_long_trend",
        "uup_ema10", "uup_sma30", "uup_sma200", "uup_short_trend", "uup_long_trend",
        "copper_close", "copper_ema10", "copper_sma30", "copper_sma200",
        "copper_short_trend", "copper_long_trend",
    ]

    print("\nlast 5 rows of new columns:")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(result[new_columns].tail(5))

    last_30 = result[new_columns].tail(30)
    nan_counts = last_30.isna().sum()
    offending = nan_counts[nan_counts > 0]
    if offending.empty:
        print("\nno NaN in last 30 rows of any new column")
    else:
        print("\nNaN found in last 30 rows for:")
        print(offending)
