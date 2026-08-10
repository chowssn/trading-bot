"""Fetches and caches historical instrument/macro data for backtesting.

Coinbase candles, yfinance macro series, and FRED credit spreads are each
cached as CSVs in backtest/data/ and reused for 24 hours before refetching.
"""

import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv
from fredapi import Fred

from backtest.etf_flows import fetch_etf_flows
from backtest.funding_rates import fetch_funding_rates

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


def _fred_observation_start(
    fred: Fred, series_id: str, start_date: datetime, extra_days: int = 365
) -> date:
    """Observation_start to request for `series_id`, extended if FRED's own
    series inception postdates `start_date`.

    Some FRED series (e.g. BAMLH0A0HYM2/BAMLC0A0CM) start later than our
    desired backtest start, or get retroactively truncated to a rolling
    window by the publisher. Requesting further back than `start_date` is
    harmless (the final reindex onto our own calendar trims it back off) and
    picks up any extra history FRED is still willing to serve.
    """
    info = fred.get_series_info(series_id)
    series_start = pd.Timestamp(info["observation_start"])
    target_start = pd.Timestamp(start_date.date())
    if series_start > target_start:
        return (start_date - timedelta(days=extra_days)).date()
    return target_start.date()


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


def _consecutive_streak(condition: pd.Series) -> pd.Series:
    """Length of the current run of consecutive True values ending at each row (0 elsewhere).

    NaN/missing input is treated as False, not as a streak-breaking special case, so the
    run-length groupby stays correct across gaps (see tests/test_indicators.py).
    """
    cond = condition.fillna(False)
    run_id = (cond != cond.shift()).cumsum()
    streak = cond.groupby(run_id).cumcount() + 1
    return streak.where(cond, 0)


def _merge_etf_flows(df: pd.DataFrame, days_back: int) -> pd.DataFrame:
    """Join BTC ETF flow columns onto `df`, gap-filled up to 3 days.

    flow_score defaults to 0.5 (neutral) rather than NaN wherever flow data is
    unavailable, so downstream signal logic always has a valid conviction modifier.
    """
    etf_flows = fetch_etf_flows(days_back=days_back)
    df = df.join(etf_flows, how="left")
    flow_cols = list(etf_flows.columns)
    df[flow_cols] = df[flow_cols].ffill(limit=3)
    df["flow_score"] = df["flow_score"].fillna(0.5)
    return df


def _merge_funding_rates(df: pd.DataFrame, days_back: int) -> pd.DataFrame:
    """Join BTC perpetual funding rate columns onto `df`, gap-filled up to 3 days.

    funding_adjustment defaults to 1.0 (neutral) rather than NaN wherever funding
    data is unavailable, so downstream position sizing always has a valid multiplier.
    """
    funding = fetch_funding_rates(days_back=days_back)
    df = df.join(funding, how="left")
    funding_cols = list(funding.columns)
    df[funding_cols] = df[funding_cols].ffill(limit=3)
    df["funding_adjustment"] = df["funding_adjustment"].fillna(1.0)
    return df


def fetch_instrument_data(
    symbol: str = "BTC-USDC", days_back: int = 1460, start_date: str | None = None
) -> pd.DataFrame:
    """Fetch daily OHLCV for `symbol` from Coinbase's public candles endpoint."""
    cache_file = DATA_DIR / f"{symbol}_ohlcv.csv"
    if _is_cache_fresh(cache_file):
        return _load_cached_csv(cache_file)

    end_time = datetime.now(timezone.utc)
    if start_date is not None:
        start_time = pd.Timestamp(start_date, tz="UTC").to_pydatetime()
    else:
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


def fetch_macro_data(days_back: int = 1460, start_date: str | None = None) -> pd.DataFrame:
    """Fetch SPY, QQQ, UUP, VIX, CPER (yfinance) and FRED HY spreads/Treasury yields.

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

    yield_curve_2s10s (DGS10 - DGS2) tracks the 2s10s Treasury curve. A negative
    curve (inversion) has historically preceded recessions; steepening back above
    -0.1 after an inversion (yield_curve_regime == 'steepening_after_inversion') is
    watched as a late-cycle warning sign rather than a bullish signal.

    Fed meeting probability tracking approximates Bloomberg WIRP functionality
    using publicly available data: FRED gives the current Fed Funds target range
    (fed_target_upper/lower/mid), and the front-month 30-day Fed Funds futures
    (ZQ=F) give the market-implied rate for month-end. The spread between the two
    (ff_implied_rate_vs_target) is converted into fed_cut_probability /
    fed_hike_probability by scaling against a 25bp hike/cut increment.
    fed_probability_4wk_change and fed_expectations_direction track the 20
    trading-day shift in cut probability: positive/​+1 means the market is pricing
    in more cuts than a month ago (dovish shift, read as bullish for BTC), negative/
    -1 means fewer cuts or more hikes (hawkish shift). For a specific upcoming
    meeting, the futures contract expiring in that meeting's month is the more
    direct read; this front-month proxy is a broad approximation, not a
    meeting-by-meeting probability curve.

    Inflation regime (inflation_regime) classifies CPI (cpi_yoy, from CPIAUCSL)
    trend into 'reaccelerating' (rising 2+ consecutive months), 'falling'
    (declining 3+ consecutive months while still above 2.5%), 'at_target' (between
    1.5% and 2.5%), 'stable_elevated' (above 2.5%, not clearly falling or rising),
    or 'below_target' (below 1.5%, not falling). PCE (pce_yoy, from PCEPI) is
    tracked alongside as the Fed's preferred inflation gauge but doesn't drive the
    regime classification. Both series are monthly, forward-filled onto the daily
    calendar.

    Fed balance sheet / Global M2 proxy: fed_balance_sheet (WALCL) and us_m2 (M2SL)
    are reindexed onto an explicit daily calendar and forward-filled before any
    EMA/SMA is computed on them, since both are far sparser (weekly/monthly) than
    the daily/business-day series above. fed_bs_short_trend (EMA10 vs SMA30) and
    fed_bs_long_trend (level vs SMA200) combine into global_m2_regime: 'expanding'
    when both agree bullish, 'contracting' when both agree bearish, else
    'transitioning'.

    HY (BAMLH0A0HYM2) and IG (BAMLC0A0CM) credit spread trend columns follow the
    same tightening-is-healthy convention as UUP: *_short_trend/*_long_trend are
    +1 when the spread is below its moving average (invert=True). credit_stress_score
    sums discrete flags (blown-out HY level, accelerating HY widening, cracking IG,
    both HY timeframes deteriorating at once) into a 0 (healthy) to 5 (stress) score.

    USO (oil_close) gets the same short/long trend treatment as copper
    (oil_short_trend: EMA10 vs SMA30, banded at 1%; oil_long_trend: close vs
    SMA200), plus an inflation-conditional oil_context_score. Long-term trend
    (SMA200) distinguishes sustained commodity cycles from short-term price
    noise. oil_context_score is capped at ±0.5 by design — it provides macro
    context only and never drives a signal on its own: rising oil confirmed by
    both trend horizons while inflation is falling/at_target reads as a growth
    positive (+0.5, or +0.25 if only the short-term trend confirms); rising oil
    while inflation is stable_elevated/reaccelerating reads as an inflation
    headwind (-0.5) regardless of the long-term trend; oil breaking down on
    both horizons reads as demand deterioration (-0.25); a short-term pullback
    within a long-term uptrend is neutral (0); all other combinations are 0.

    BTC spot ETF flows (see backtest/etf_flows.py) are joined in last and
    gap-filled up to 3 days; flow_score falls back to 0.5 (neutral) rather than
    NaN wherever flow data is unavailable so downstream signal logic always has
    a valid conviction modifier.

    BTC perpetual funding rates (see backtest/funding_rates.py) are joined in
    last and gap-filled up to 3 days; funding_adjustment falls back to 1.0
    (neutral) rather than NaN wherever funding data is unavailable.
    """
    cache_file = DATA_DIR / f"macro_v2_{days_back}d.csv"
    if _is_cache_fresh(cache_file):
        return _load_cached_csv(cache_file)

    end_date = datetime.now(timezone.utc)
    if start_date is not None:
        start_date = pd.Timestamp(start_date, tz="UTC").to_pydatetime()
    else:
        start_date = end_date - timedelta(days=days_back)

    tickers = {
        "SPY": "spy_close",
        "QQQ": "qqq_close",
        "UUP": "uup_close",
        "^VIX": "vix_close",
        "CPER": "copper_close",
        "USO": "oil_close",
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

    oil_ema10 = _ema(prices["oil_close"], 10).rename("oil_ema10")
    oil_sma30 = _sma(prices["oil_close"], 30).rename("oil_sma30")
    oil_sma200 = _sma(prices["oil_close"], 200).rename("oil_sma200")
    oil_short_trend = _trend_banded(oil_ema10, oil_sma30, band=0.01).rename("oil_short_trend")
    oil_long_trend = _trend_binary(prices["oil_close"], oil_sma200).rename("oil_long_trend")

    load_dotenv()
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("FRED_API_KEY not set in .env")

    fred = Fred(api_key=api_key)

    # WALCL (weekly) and M2SL (monthly) are far sparser than the daily/business-day
    # FRED series above, so a plain .ffill() on the raw series wouldn't produce daily
    # values — reindex onto an explicit daily calendar first, per fed_balance_sheet/
    # us_m2 being defined as the forward-filled-to-daily series.
    daily_calendar = pd.date_range(
        start=start_date.replace(hour=0, minute=0, second=0, microsecond=0),
        end=end_date.replace(hour=0, minute=0, second=0, microsecond=0),
        freq="D",
        tz="UTC",
    )

    walcl = fred.get_series(
        "WALCL",
        observation_start=start_date.date(),
        observation_end=end_date.date(),
    )
    walcl.index = _normalize_to_utc_dates(walcl.index)
    fed_balance_sheet = walcl.reindex(daily_calendar).ffill().rename("fed_balance_sheet")

    m2sl = fred.get_series(
        "M2SL",
        observation_start=start_date.date(),
        observation_end=end_date.date(),
    )
    m2sl.index = _normalize_to_utc_dates(m2sl.index)
    us_m2 = m2sl.reindex(daily_calendar).ffill().rename("us_m2")

    fed_bs_ema10 = _ema(fed_balance_sheet, 10).rename("fed_bs_ema10")
    fed_bs_sma30 = _sma(fed_balance_sheet, 30).rename("fed_bs_sma30")
    fed_bs_sma200 = _sma(fed_balance_sheet, 200).rename("fed_bs_sma200")
    fed_bs_short_trend = _trend_binary(fed_bs_ema10, fed_bs_sma30).rename("fed_bs_short_trend")
    fed_bs_long_trend = _trend_binary(fed_balance_sheet, fed_bs_sma200).rename(
        "fed_bs_long_trend"
    )
    fed_bs_13wk_change_pct = (fed_balance_sheet.pct_change(13 * 7) * 100).rename(
        "fed_bs_13wk_change_pct"
    )

    global_m2_regime = pd.Series(
        np.select(
            [
                (fed_bs_short_trend == 1) & (fed_bs_long_trend == 1),
                (fed_bs_short_trend == -1) & (fed_bs_long_trend == -1),
            ],
            ["expanding", "contracting"],
            default="transitioning",
        ),
        index=fed_bs_short_trend.index,
    ).mask(fed_bs_short_trend.isna() | fed_bs_long_trend.isna())
    global_m2_regime.name = "global_m2_regime"

    hy_spread = fred.get_series(
        "BAMLH0A0HYM2",
        observation_start=_fred_observation_start(fred, "BAMLH0A0HYM2", start_date),
        observation_end=end_date.date(),
    )
    hy_spread.index = _normalize_to_utc_dates(hy_spread.index)
    hy_spread.name = "hy_spread"
    hy_spread = hy_spread.ffill(limit=7)
    print(f"hy_spread first non-null date: {hy_spread.dropna().index.min()}")

    hy_spread_ema10 = _ema(hy_spread, 10).rename("hy_spread_ema10")
    hy_spread_sma30 = _sma(hy_spread, 30).rename("hy_spread_sma30")
    hy_spread_sma200 = _sma(hy_spread, 200).rename("hy_spread_sma200")
    hy_spread_short_trend = _trend_binary(hy_spread_ema10, hy_spread_sma30, invert=True).rename(
        "hy_spread_short_trend"
    )
    hy_spread_long_trend = _trend_binary(hy_spread, hy_spread_sma200, invert=True).rename(
        "hy_spread_long_trend"
    )
    hy_spread_5d_change = hy_spread.diff(5).rename("hy_spread_5d_change")
    hy_spread_acceleration = (hy_spread_5d_change - hy_spread_5d_change.shift(5)).rename(
        "hy_spread_acceleration"
    )

    ig_spread = fred.get_series(
        "BAMLC0A0CM",
        observation_start=_fred_observation_start(fred, "BAMLC0A0CM", start_date),
        observation_end=end_date.date(),
    )
    ig_spread.index = _normalize_to_utc_dates(ig_spread.index)
    ig_spread.name = "ig_spread"
    ig_spread = ig_spread.ffill(limit=7)
    print(f"ig_spread first non-null date: {ig_spread.dropna().index.min()}")

    ig_spread_ema10 = _ema(ig_spread, 10).rename("ig_spread_ema10")
    ig_spread_sma30 = _sma(ig_spread, 30).rename("ig_spread_sma30")
    ig_spread_sma200 = _sma(ig_spread, 200).rename("ig_spread_sma200")
    ig_spread_short_trend = _trend_binary(ig_spread_ema10, ig_spread_sma30, invert=True).rename(
        "ig_spread_short_trend"
    )
    ig_spread_long_trend = _trend_binary(ig_spread, ig_spread_sma200, invert=True).rename(
        "ig_spread_long_trend"
    )
    ig_spread_5d_change = ig_spread.diff(5).rename("ig_spread_5d_change")

    credit_stress_score = pd.Series(
        np.where(hy_spread > 600, 2, 0)
        + np.where(hy_spread > 450, 1, 0)
        + np.where(hy_spread_acceleration > 0.2, 1, 0)
        + np.where(ig_spread_5d_change > 0.1, 1, 0)
        + np.where((hy_spread_long_trend == -1) & (hy_spread_short_trend == -1), 1, 0),
        index=hy_spread.index,
    ).rename("credit_stress_score")

    dgs2 = fred.get_series(
        "DGS2",
        observation_start=start_date.date(),
        observation_end=end_date.date(),
    )
    dgs2.index = _normalize_to_utc_dates(dgs2.index)
    dgs2.name = "dgs2"
    dgs2 = dgs2.ffill()

    dgs10 = fred.get_series(
        "DGS10",
        observation_start=start_date.date(),
        observation_end=end_date.date(),
    )
    dgs10.index = _normalize_to_utc_dates(dgs10.index)
    dgs10.name = "dgs10"
    dgs10 = dgs10.ffill()

    yield_curve_2s10s = (dgs10 - dgs2).rename("yield_curve_2s10s")

    curve_change_5d = yield_curve_2s10s.diff(5)
    yield_curve_direction = pd.Series(
        np.where(
            curve_change_5d > 0.05, 1, np.where(curve_change_5d < -0.05, -1, 0)
        ),
        index=yield_curve_2s10s.index,
        dtype=float,
    ).mask(curve_change_5d.isna())
    yield_curve_direction.name = "yield_curve_direction"

    curve_20d_ago = yield_curve_2s10s.shift(20)
    regime_conditions = [
        (curve_20d_ago < -0.1) & (yield_curve_2s10s >= -0.1),
        yield_curve_2s10s > 0.1,
        yield_curve_2s10s < -0.1,
    ]
    regime_choices = ["steepening_after_inversion", "normal", "inverted"]
    yield_curve_regime = pd.Series(
        np.select(regime_conditions, regime_choices, default="flat"),
        index=yield_curve_2s10s.index,
    ).mask(yield_curve_2s10s.isna())
    yield_curve_regime.name = "yield_curve_regime"

    dff = fred.get_series(
        "DFF",
        observation_start=start_date.date(),
        observation_end=end_date.date(),
    )
    dff.index = _normalize_to_utc_dates(dff.index)
    dff.name = "fed_funds_effective"
    dff = dff.ffill()

    dfedtaru = fred.get_series(
        "DFEDTARU",
        observation_start=start_date.date(),
        observation_end=end_date.date(),
    )
    dfedtaru.index = _normalize_to_utc_dates(dfedtaru.index)
    dfedtaru.name = "fed_target_upper"
    dfedtaru = dfedtaru.ffill()

    dfedtarl = fred.get_series(
        "DFEDTARL",
        observation_start=start_date.date(),
        observation_end=end_date.date(),
    )
    dfedtarl.index = _normalize_to_utc_dates(dfedtarl.index)
    dfedtarl.name = "fed_target_lower"
    dfedtarl = dfedtarl.ffill()

    fed_target_mid = ((dfedtaru + dfedtarl) / 2).rename("fed_target_mid")

    ff_hist = yf.Ticker("ZQ=F").history(
        start=start_date.date(), end=end_date.date() + timedelta(days=1)
    )
    ff_futures_price = ff_hist["Close"].rename("ff_futures_price")
    ff_futures_price.index = _normalize_to_utc_dates(ff_futures_price.index)

    fed_block = pd.concat(
        [dff, dfedtaru, dfedtarl, fed_target_mid, ff_futures_price], axis=1, sort=True
    ).sort_index()
    fed_block = fed_block.ffill()

    ff_implied_rate = (100 - fed_block["ff_futures_price"]).rename("ff_implied_rate")
    ff_implied_rate_vs_target = (ff_implied_rate - fed_block["fed_target_mid"]).rename(
        "ff_implied_rate_vs_target"
    )

    fed_cut_probability = (
        ((fed_block["fed_target_mid"] - ff_implied_rate) / 0.25)
        .clip(0, 1)
        .rename("fed_cut_probability")
    )
    fed_hike_probability = (
        ((ff_implied_rate - fed_block["fed_target_mid"]) / 0.25)
        .clip(0, 1)
        .rename("fed_hike_probability")
    )

    fed_probability_4wk_change = (
        fed_cut_probability - fed_cut_probability.shift(20)
    ).rename("fed_probability_4wk_change")

    fed_expectations_direction = pd.Series(
        np.select(
            [fed_probability_4wk_change > 0.10, fed_probability_4wk_change < -0.10],
            [1, -1],
            default=0,
        ),
        index=fed_probability_4wk_change.index,
        dtype=float,
    ).mask(fed_probability_4wk_change.isna())
    fed_expectations_direction.name = "fed_expectations_direction"

    # cpi_yoy/pce_yoy need 12 months of history before start_date to compute a
    # valid year-over-year change there, so fetch with an extra lookback buffer;
    # the final reindex onto full_index trims anything before start_date back off.
    inflation_lookback_start = (start_date - timedelta(days=400)).date()

    cpi = fred.get_series(
        "CPIAUCSL",
        observation_start=inflation_lookback_start,
        observation_end=end_date.date(),
    )
    cpi.index = _normalize_to_utc_dates(cpi.index)

    pce = fred.get_series(
        "PCEPI",
        observation_start=inflation_lookback_start,
        observation_end=end_date.date(),
    )
    pce.index = _normalize_to_utc_dates(pce.index)

    cpi_yoy = (cpi.pct_change(12) * 100).rename("cpi_yoy")
    pce_yoy = (pce.pct_change(12) * 100).rename("pce_yoy")

    cpi_yoy_diff = cpi_yoy.diff()
    rising_streak = _consecutive_streak(cpi_yoy_diff > 0)
    falling_streak = _consecutive_streak(cpi_yoy_diff < 0)

    inflation_conditions = [
        rising_streak >= 2,
        (falling_streak >= 3) & (cpi_yoy > 2.5),
        (cpi_yoy >= 1.5) & (cpi_yoy <= 2.5),
        cpi_yoy > 2.5,
    ]
    inflation_choices = ["reaccelerating", "falling", "at_target", "stable_elevated"]
    inflation_regime = pd.Series(
        np.select(inflation_conditions, inflation_choices, default="below_target"),
        index=cpi_yoy.index,
    ).mask(cpi_yoy.isna())
    inflation_regime.name = "inflation_regime"

    df = pd.concat(
        [
            prices,
            spy_ema10, spy_sma30, spy_sma200, spy_short_trend, spy_long_trend,
            qqq_ema10, qqq_sma30, qqq_sma200, qqq_short_trend, qqq_long_trend,
            qqq_spy_ratio, qqq_spy_ratio_ema10, qqq_spy_ratio_sma30, qqq_spy_ratio_sma200,
            qqq_spy_short_trend, qqq_spy_long_trend,
            uup_ema10, uup_sma30, uup_sma200, uup_short_trend, uup_long_trend,
            copper_ema10, copper_sma30, copper_sma200, copper_short_trend, copper_long_trend,
            oil_ema10, oil_sma30, oil_sma200, oil_short_trend, oil_long_trend,
            fed_balance_sheet, us_m2,
            fed_bs_ema10, fed_bs_sma30, fed_bs_sma200, fed_bs_short_trend, fed_bs_long_trend,
            fed_bs_13wk_change_pct, global_m2_regime,
            hy_spread,
            hy_spread_ema10, hy_spread_sma30, hy_spread_sma200,
            hy_spread_short_trend, hy_spread_long_trend,
            hy_spread_5d_change, hy_spread_acceleration,
            ig_spread, ig_spread_ema10, ig_spread_sma30, ig_spread_sma200,
            ig_spread_short_trend, ig_spread_long_trend, ig_spread_5d_change,
            credit_stress_score,
            dgs2, dgs10,
            yield_curve_2s10s, yield_curve_direction, yield_curve_regime,
            dff, dfedtaru, dfedtarl, fed_target_mid,
            ff_futures_price, ff_implied_rate, ff_implied_rate_vs_target,
            fed_cut_probability, fed_hike_probability,
            fed_probability_4wk_change, fed_expectations_direction,
            cpi_yoy, pce_yoy, inflation_regime,
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

    # oil_context_score mixes oil's daily trend columns with the monthly
    # inflation_regime, so it's computed here once both are on the same
    # forward-filled daily calendar rather than at their native, mismatched
    # frequencies.
    oil_context_score = pd.Series(
        np.select(
            [
                (df["oil_short_trend"] == 1)
                & (df["oil_long_trend"] == 1)
                & df["inflation_regime"].isin(["falling", "at_target"]),
                (df["oil_short_trend"] == 1)
                & (df["oil_long_trend"] == -1)
                & df["inflation_regime"].isin(["falling", "at_target"]),
                (df["oil_short_trend"] == 1)
                & df["inflation_regime"].isin(["stable_elevated", "reaccelerating"]),
                (df["oil_short_trend"] == -1) & (df["oil_long_trend"] == -1),
                (df["oil_short_trend"] == -1) & (df["oil_long_trend"] == 1),
            ],
            [0.5, 0.25, -0.5, -0.25, 0.0],
            default=0.0,
        ),
        index=df.index,
    ).mask(
        df["oil_short_trend"].isna() | df["oil_long_trend"].isna() | df["inflation_regime"].isna()
    )
    oil_context_score.name = "oil_context_score"
    df["oil_context_score"] = oil_context_score

    df = _merge_etf_flows(df, days_back)
    df = _merge_funding_rates(df, days_back)

    _save_cache(df, cache_file)
    return df


def fetch_all(
    symbol: str = "BTC-USDC", days_back: int = 1460, start_date: str | None = None
) -> pd.DataFrame:
    """Fetch instrument OHLCV and macro data and merge into one frame."""
    instrument = fetch_instrument_data(symbol, days_back, start_date)
    macro = fetch_macro_data(days_back, start_date)

    combined = instrument.join(macro, how="outer")
    combined = combined.sort_index().ffill()
    combined = combined.dropna(subset=["close"])
    return combined


def _fmt_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    text = str(value)
    return text if len(text) <= 20 else text[:17] + "..."


_SMA200_DERIVED_COLS = frozenset({"global_m2_regime", "oil_context_score"})


def _null_pct_threshold(col: str) -> float:
    """Null% warning threshold, widened for columns with expected startup NaN.

    Rolling-window columns can't produce a value until the window fills, so
    a flat 5% threshold flags them as false positives on every run:
    - *_sma200 / *_long_trend: up to 200/730 days of startup NaN is normal.
    - *_sma30: up to 30 days of startup NaN is normal.
    - fed_bs_13wk_change_pct (90-day pct_change window): ditto at 90 days.
    - global_m2_regime / oil_context_score: not named *_long_trend themselves,
      but each is masked null wherever the *_long_trend column it's gated on
      is null, so they inherit that column's 200-day startup window too.
    Everything else keeps the original 5% threshold.
    """
    if col.endswith("_sma200") or col.endswith("_long_trend") or col in _SMA200_DERIVED_COLS:
        return 22.0
    if col.endswith("_sma30"):
        return 6.0
    if col == "fed_bs_13wk_change_pct":
        return 8.0
    return 5.0


# Columns (or column-name prefixes) with a known, understood data limitation.
# A column matching an entry here is reported as INFO instead of WARNING and
# does not count toward FAIL — the underlying cause is documented, expected,
# and not a signal of a genuine data quality regression. See the "Data
# Maintenance" section of CLAUDE.md for the operational detail behind each.
_KNOWN_LIMITATIONS: list[dict] = [
    {
        "prefixes": ("hy_spread", "ig_spread"),
        "exact": frozenset({"credit_stress_score"}),
        "reason": (
            "FRED series truncated to 3-year rolling window since April 2026, "
            "data unavailable before 2023-07-11"
        ),
    },
    {
        "prefixes": ("funding",),
        "exact": frozenset({"fundingRate"}),
        "reason": "Binance/Bybit geo-blocked in this environment, defaults to neutral 1.0",
    },
    {
        "prefixes": (),
        "exact": frozenset({
            "ibit_flow_musd", "fbtc_flow_musd", "arkb_flow_musd",
            "bitb_flow_musd", "gbtc_flow_musd", "ex_gbtc_flow_musd",
        }),
        "reason": "yfinance does not serve BTC ETF shares outstanding, per CLAUDE.md",
    },
    {
        "prefixes": ("flow_",),
        "exact": frozenset({"daily_flow_musd", "ex_gbtc_flow_score", "gbtc_outflow_pressure"}),
        "reason": (
            "ETF data only available from January 2024 (BTC ETF launch date), "
            "pre-launch rows are correctly null"
        ),
    },
]


def _known_limitation_reason(col: str) -> str | None:
    """Return the documented reason `col` is a known limitation, or None."""
    for entry in _KNOWN_LIMITATIONS:
        if col in entry["exact"] or col.startswith(entry["prefixes"]):
            return entry["reason"]
    return None


def verify_data_quality(df: pd.DataFrame) -> None:
    """Print a data-quality report for `df` and a final PASS/FAIL summary.

    Every column is reported (dtype, null count/%, min, max). A column is
    flagged WARNING if its null% exceeds its threshold (see
    `_null_pct_threshold`; 5% by default, widened for known rolling-window
    columns), or if it holds a value considered implausible for its kind
    (negative prices, >20% yields, >2000bps credit spreads, or a funding
    rate outside -0.5%/+0.5%) — unless the column matches `_KNOWN_LIMITATIONS`,
    in which case it's printed as INFO instead. PASS requires zero warnings;
    INFO messages don't count toward FAIL.
    """
    warnings: list[str] = []
    infos: list[str] = []
    total_rows = len(df)

    if isinstance(df.index, pd.DatetimeIndex) and total_rows:
        date_range = f"{df.index.min()} to {df.index.max()}"
    else:
        date_range = "n/a"

    print(f"Total rows: {total_rows}")
    print(f"Date range: {date_range}")
    print()

    header = f"{'column':<35}{'dtype':<10}{'nulls':>8}{'null %':>9}{'min':>22}{'max':>22}"
    print(header)
    print("-" * len(header))

    for col in df.columns:
        series = df[col]
        null_count = int(series.isna().sum())
        null_pct = (null_count / total_rows * 100) if total_rows else 0.0

        non_null = series.dropna()
        try:
            col_min = non_null.min() if not non_null.empty else "n/a"
            col_max = non_null.max() if not non_null.empty else "n/a"
        except TypeError:
            col_min, col_max = "n/a", "n/a"

        print(
            f"{col:<35}{str(series.dtype):<10}{null_count:>8}{null_pct:>8.1f}%"
            f"{_fmt_value(col_min):>22}{_fmt_value(col_max):>22}"
        )

        threshold = _null_pct_threshold(col)
        if null_pct > threshold:
            reason = _known_limitation_reason(col)
            if reason:
                infos.append(
                    f"INFO: '{col}' has {null_pct:.1f}% null values (> {threshold:.0f}%) "
                    f"— known limitation: {reason}"
                )
            else:
                warnings.append(
                    f"WARNING: '{col}' has {null_pct:.1f}% null values (> {threshold:.0f}%)"
                )

    print()

    negative_price_cols = ["close", "spy_close", "qqq_close", "copper_close", "oil_close"]
    for col in negative_price_cols:
        if col in df.columns:
            n_bad = int((df[col] < 0).sum())
            if n_bad:
                warnings.append(f"WARNING: '{col}' has {n_bad} negative value(s)")

    yield_cols = ["dgs2", "dgs10"]
    for col in yield_cols:
        if col in df.columns:
            n_bad = int((df[col] > 20).sum())
            if n_bad:
                warnings.append(f"WARNING: '{col}' has {n_bad} value(s) > 20% yield")

    spread_cols = ["hy_spread", "ig_spread"]
    for col in spread_cols:
        if col in df.columns:
            n_bad = int((df[col] > 2000).sum())
            if n_bad:
                warnings.append(f"WARNING: '{col}' has {n_bad} value(s) > 2000bps spread")

    # funding rate is exposed either as a percentage (funding_rate_pct, e.g.
    # 0.01 = 0.01%) or a raw fraction (fundingRate, e.g. 0.0001); check
    # whichever is present against the equivalent -0.5%/+0.5% bound.
    if "funding_rate_pct" in df.columns:
        n_bad = int((df["funding_rate_pct"].abs() > 0.5).sum())
        if n_bad:
            warnings.append(
                f"WARNING: 'funding_rate_pct' has {n_bad} value(s) outside -0.5%/+0.5%"
            )
    if "fundingRate" in df.columns:
        n_bad = int((df["fundingRate"].abs() > 0.005).sum())
        if n_bad:
            warnings.append(
                f"WARNING: 'fundingRate' has {n_bad} value(s) outside -0.5%/+0.5%"
            )

    for i in infos:
        print(i)
    if infos:
        print()

    if warnings:
        for w in warnings:
            print(w)
        print()
        print(f"FAIL ({len(warnings)} warning(s))")
    else:
        print("PASS")


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
        "oil_close", "oil_ema10", "oil_sma30", "oil_sma200",
        "oil_short_trend", "oil_long_trend",
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

    print("\nyield_curve_regime (last 30 days):")
    print(result["yield_curve_regime"].tail(30))

    print("\nfed_cut_probability and fed_expectations_direction (last 60 days):")
    print(result[["fed_cut_probability", "fed_expectations_direction"]].tail(60))

    print("\ninflation_regime (full date range):")
    print(result["inflation_regime"])

    print("\nglobal_m2_regime (full date range):")
    print(result["global_m2_regime"])

    print("\ncredit_stress_score (last 60 days):")
    print(result["credit_stress_score"].tail(60))

    print("\noil_context_score, oil_long_trend, inflation_regime (last 60 days):")
    print(result[["oil_context_score", "oil_long_trend", "inflation_regime"]].tail(60))

    new_sma200_columns = ["fed_bs_sma200", "hy_spread_sma200", "ig_spread_sma200"]
    print("\nnew SMA200 columns (last 5 rows):")
    print(result[new_sma200_columns].tail(5))
    new_sma200_nan_counts = result[new_sma200_columns].tail(5).isna().sum()
    offending_sma200 = new_sma200_nan_counts[new_sma200_nan_counts > 0]
    if offending_sma200.empty:
        print("\nno NaN in last 5 rows of any new SMA200 column")
    else:
        print("\nNaN found in last 5 rows for:")
        print(offending_sma200)

    # DGS2/DGS10 and CPI/Fed target series go back decades on FRED; pull a longer
    # window than the default 1460-day backtest lookback so the 2022-2023 inversion,
    # hiking cycle, and inflation peak/decline are all covered.
    long_macro = fetch_macro_data(days_back=1700)
    print("\nyield_curve_regime spot-check, 2022-01-01 to 2023-12-31:")
    print(long_macro.loc["2022-01-01":"2023-12-31", "yield_curve_regime"].value_counts())

    print("\ninflation_regime spot-check, 2022 (expect elevated/reaccelerating CPI):")
    print(long_macro.loc["2022-01-01":"2022-12-31", "inflation_regime"].value_counts())
    print("\nfed_expectations_direction spot-check, 2022 (expect hawkish/-1 tilt):")
    print(long_macro.loc["2022-01-01":"2022-12-31", "fed_expectations_direction"].value_counts())

    print("\ninflation_regime spot-check, late 2023 (expect falling/at_target CPI):")
    print(long_macro.loc["2023-07-01":"2023-12-31", "inflation_regime"].value_counts())
    print("\nfed_expectations_direction spot-check, late 2023 (expect dovish/+1 shift):")
    print(long_macro.loc["2023-07-01":"2023-12-31", "fed_expectations_direction"].value_counts())

    print("\ndata quality report:")
    verify_data_quality(result)
