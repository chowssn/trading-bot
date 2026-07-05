"""Technical indicators for backtesting.

Pure pandas/numpy implementations only — no ta or talib — so the math
stays transparent and auditable.
"""

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI using Wilder smoothing."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range using Wilder smoothing."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.ewm(alpha=1 / period, adjust=False).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index."""
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr_smooth = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=1 / period, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=1 / period, adjust=False).mean()

    plus_di = 100 * (plus_dm_smooth / atr_smooth)
    minus_di = 100 * (minus_dm_smooth / atr_smooth)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def volume_sma(volume: pd.Series, period: int = 20) -> pd.Series:
    """SMA of volume."""
    return volume.rolling(window=period).mean()


if __name__ == "__main__":
    from backtest.data_fetcher import fetch_btc_candles

    df = fetch_btc_candles()

    results = {
        "ema": ema(df["close"], 20),
        "sma": sma(df["close"], 20),
        "rsi": rsi(df["close"]),
        "atr": atr(df["high"], df["low"], df["close"]),
        "adx": adx(df["high"], df["low"], df["close"]),
        "volume_sma": volume_sma(df["volume"]),
    }

    for name, series in results.items():
        print(f"\n{name} (last 5 rows):")
        print(series.tail(5))
