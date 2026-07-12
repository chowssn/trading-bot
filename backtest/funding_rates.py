"""Fetches and caches BTC perpetual funding rates from Binance's public API.

Standalone module: no dependency on backtest/data_fetcher.py, no authentication
required. Binance publishes a new funding rate every 8 hours; derived rolling
averages are computed on that native 8-hourly series before the frame is
resampled down to one row per UTC day.

If Binance responds with a 4xx error (e.g. geo-blocking), Bybit's funding
history endpoint is tried as a secondary source before giving up and falling
back to the on-disk cache.
"""

import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CACHE_FILE = DATA_DIR / "btc_funding_rates.csv"
CACHE_TTL_SECONDS = 8 * 60 * 60

FUNDING_RATE_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
PAGE_LIMIT = 1000

BYBIT_FUNDING_RATE_URL = "https://api.bybit.com/v5/market/funding/history"
BYBIT_PAGE_LIMIT = 200

FUNDING_INTERVAL_HOURS = 8

EXTREME_LONG_THRESHOLD = 0.0005
EXTREME_SHORT_THRESHOLD = -0.0002

FUNDING_ADJUSTMENT = {
    "extreme_long": 0.80,
    "extreme_short": 1.10,
    "normal": 1.00,
}

FUNDING_COLUMNS = [
    "fundingRate",
    "funding_rate_pct",
    "funding_7d_avg",
    "funding_30d_avg",
    "funding_vs_30d_baseline",
    "funding_regime",
    "funding_adjustment",
]


def _is_cache_fresh(path: Path) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < CACHE_TTL_SECONDS


def _load_cached_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.index.name = "date"
    return df


def _save_cache(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path)


def _empty_funding_frame() -> pd.DataFrame:
    df = pd.DataFrame(columns=FUNDING_COLUMNS)
    df.index.name = "date"
    return df


def _fetch_raw_funding_rates_binance(
    symbol: str, start_time: datetime, end_time: datetime
) -> pd.DataFrame:
    """Paginate backwards from `end_time` in `PAGE_LIMIT`-record chunks until `start_time` is covered.

    Binance returns each page in ascending fundingTime order. Each successive
    request's endTime is set to just before the oldest record of the previous
    page, so pages don't overlap.
    """
    start_ms = int(start_time.timestamp() * 1000)
    chunk_end_ms = int(end_time.timestamp() * 1000)

    all_records = []
    while True:
        params = {"symbol": symbol, "endTime": chunk_end_ms, "limit": PAGE_LIMIT}
        response = requests.get(FUNDING_RATE_URL, params=params, timeout=15)
        response.raise_for_status()
        page = response.json()
        if not page:
            break

        all_records = page + all_records
        oldest_time = int(page[0]["fundingTime"])
        if oldest_time <= start_ms or len(page) < PAGE_LIMIT:
            break

        chunk_end_ms = oldest_time - 1
        time.sleep(0.2)

    if not all_records:
        raise ValueError(f"No funding rate data returned from Binance for {symbol}")

    df = pd.DataFrame(all_records)
    df["fundingTime"] = pd.to_datetime(df["fundingTime"].astype(np.int64), unit="ms", utc=True)
    df["fundingRate"] = df["fundingRate"].astype(float)
    df = (
        df[["fundingTime", "fundingRate"]]
        .sort_values("fundingTime")
        .drop_duplicates(subset="fundingTime")
        .set_index("fundingTime")
    )
    return df


def _fetch_raw_funding_rates_bybit(symbol: str, start_time: datetime) -> pd.DataFrame:
    """Paginate backwards from the most recent funding record using Bybit's cursor.

    Bybit returns each page in descending fundingRateTimestamp order (newest
    first). `nextPageCursor` on the response is echoed back as the `cursor`
    param to fetch the next (older) page, until either the page's oldest
    record reaches `start_time` or the API stops returning a cursor.
    """
    start_ms = int(start_time.timestamp() * 1000)

    all_records = []
    cursor = ""
    while True:
        params = {"category": "linear", "symbol": symbol, "limit": BYBIT_PAGE_LIMIT}
        if cursor:
            params["cursor"] = cursor

        response = requests.get(BYBIT_FUNDING_RATE_URL, params=params, timeout=15)
        response.raise_for_status()
        result = response.json().get("result", {})
        page = result.get("list", [])
        if not page:
            break

        all_records.extend(page)
        oldest_time = int(page[-1]["fundingRateTimestamp"])
        cursor = result.get("nextPageCursor", "")
        if oldest_time <= start_ms or not cursor:
            break

        time.sleep(0.2)

    if not all_records:
        raise ValueError(f"No funding rate data returned from Bybit for {symbol}")

    df = pd.DataFrame(all_records)
    df["fundingTime"] = pd.to_datetime(
        df["fundingRateTimestamp"].astype(np.int64), unit="ms", utc=True
    )
    df["fundingRate"] = df["fundingRate"].astype(float)
    df = (
        df[["fundingTime", "fundingRate"]]
        .sort_values("fundingTime")
        .drop_duplicates(subset="fundingTime")
        .set_index("fundingTime")
    )
    return df


def _is_4xx_error(exc: Exception) -> bool:
    return (
        isinstance(exc, requests.exceptions.HTTPError)
        and exc.response is not None
        and 400 <= exc.response.status_code < 500
    )


def _fetch_raw_funding_rates(
    symbol: str, start_time: datetime, end_time: datetime
) -> pd.DataFrame:
    """Fetch raw 8-hourly funding rates, preferring Binance.

    Falls back to Bybit only when Binance responds with a 4xx error (e.g.
    geo-blocking). Other failures (timeouts, 5xx, connection errors) propagate
    so the cache fallback in fetch_funding_rates() can take over instead.
    """
    try:
        return _fetch_raw_funding_rates_binance(symbol, start_time, end_time)
    except requests.exceptions.HTTPError as exc:
        if not _is_4xx_error(exc):
            raise
        logger.warning(
            "Binance funding rate fetch returned %s; falling back to Bybit",
            exc.response.status_code,
        )
        return _fetch_raw_funding_rates_bybit(symbol, start_time)


def _add_funding_derived_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """funding_rate_pct and rolling 7d/30d averages, computed on the native 8-hourly series."""
    df = raw.copy()
    df["funding_rate_pct"] = df["fundingRate"] * 100
    df["funding_7d_avg"] = df["fundingRate"].rolling(21).mean()
    df["funding_30d_avg"] = df["fundingRate"].rolling(90).mean()
    df["funding_vs_30d_baseline"] = df["funding_7d_avg"] - df["funding_30d_avg"]
    return df


def _resample_daily(df: pd.DataFrame) -> pd.DataFrame:
    daily = df.resample("D").last()
    daily.index.name = "date"
    return daily


def _classify_regime(funding_7d_avg: pd.Series) -> pd.DataFrame:
    """funding_regime/funding_adjustment from funding_7d_avg; NaN input falls back to neutral (1.0)."""
    funding_regime = pd.Series(
        np.select(
            [funding_7d_avg > EXTREME_LONG_THRESHOLD, funding_7d_avg < EXTREME_SHORT_THRESHOLD],
            ["extreme_long", "extreme_short"],
            default="normal",
        ),
        index=funding_7d_avg.index,
    ).mask(funding_7d_avg.isna())
    funding_regime.name = "funding_regime"

    funding_adjustment = (
        funding_regime.map(FUNDING_ADJUSTMENT).fillna(1.0).rename("funding_adjustment")
    )

    return pd.concat([funding_regime, funding_adjustment], axis=1)


def _add_regime_columns(daily: pd.DataFrame) -> pd.DataFrame:
    regime = _classify_regime(daily["funding_7d_avg"])
    return pd.concat([daily, regime], axis=1)


def fetch_funding_rates(symbol: str = "BTCUSDT", days_back: int = 1460) -> pd.DataFrame:
    """Fetch BTC perpetual funding rates with derived rolling averages and regime.

    Raw 8-hourly funding rates are paginated from Binance, derived rolling
    averages (funding_7d_avg, funding_30d_avg) are computed on that native
    frequency, then the frame is resampled to one row per UTC day (last value
    of the day) before funding_regime/funding_adjustment are classified.

    On any fetch failure: logs a warning and falls back to the cached data if
    available; if no cache exists either, returns an empty DataFrame with the
    correct columns rather than raising.
    """
    try:
        if _is_cache_fresh(CACHE_FILE):
            return _load_cached_csv(CACHE_FILE)

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days_back)

        raw = _fetch_raw_funding_rates(symbol, start_time, end_time)
        raw = _add_funding_derived_columns(raw)
        daily = _resample_daily(raw)
        daily = _add_regime_columns(daily)

        _save_cache(daily, CACHE_FILE)
        return daily
    except Exception as exc:
        logger.warning("Funding rate fetch failed (%s); falling back to cache", exc)
        if not CACHE_FILE.exists():
            return _empty_funding_frame()
        return _load_cached_csv(CACHE_FILE)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = fetch_funding_rates()

    print(f"shape: {result.shape}")
    if not result.empty:
        print(f"date range: {result.index.min()} to {result.index.max()}")

    print("\nlast 10 rows:")
    display_cols = ["funding_7d_avg", "funding_regime", "funding_adjustment"]
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(result[display_cols].tail(10))

    if not result.empty:
        latest = result.iloc[-1]
        print(f"\ncurrent funding_regime: {latest['funding_regime']}")
        print(f"current funding_adjustment: {latest['funding_adjustment']}")

        print(f"\nmin funding_rate seen: {result['fundingRate'].min()}")
        print(f"max funding_rate seen: {result['fundingRate'].max()}")
