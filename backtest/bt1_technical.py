"""BT1 — technical-only BTC long/flat strategy.

No macro filter, no regime detector: entries and exits are driven purely
by price/volume technical indicators computed on daily BTC OHLCV.
"""

import numpy as np
import pandas as pd

from backtest.data_fetcher import fetch_instrument_data
from backtest.engine import BacktestEngine
from backtest.indicators import adx, atr, ema, rsi, sma, volume_sma

ATR_STOP_MULTIPLIER = 2
ADX_TREND_THRESHOLD = 25
RSI_OVEREXTENDED_THRESHOLD = 75


def build_signals(data: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Generate the signal, position-size, and stop series for BT1.

    Entry (all must hold the same day): EMA(8) > EMA(21), close above
    both SMA(50) and SMA(200), ADX(14) > 25, volume above its 20-day
    average, and RSI(14) < 75.

    Exit (any one, while in a trade): EMA(8) crosses below EMA(21),
    close falls below SMA(50), or close breaches a stop set at entry to
    entry_price - 2xATR(14). `entry_price` is approximated as the close
    on the day the signal fires, since BacktestEngine itself applies the
    next-day-open fill — the exact fill price isn't known at signal time.

    Returns:
        (signals, position_sizes, stops) Series aligned to `data.index`.
    """
    close = data["close"]
    volume = data["volume"]

    ema8 = ema(close, 8)
    ema21 = ema(close, 21)
    sma50 = sma(close, 50)
    sma200 = sma(close, 200)
    adx14 = adx(data["high"], data["low"], close, 14)
    atr14 = atr(data["high"], data["low"], close, 14)
    vol_sma20 = volume_sma(volume, 20)
    rsi14 = rsi(close, 14)

    entry_conditions = (
        (ema8 > ema21)
        & (close > sma50)
        & (close > sma200)
        & (adx14 > ADX_TREND_THRESHOLD)
        & (volume > vol_sma20)
        & (rsi14 < RSI_OVEREXTENDED_THRESHOLD)
    )

    signals = pd.Series(0, index=data.index, dtype=int)
    stops = pd.Series(np.nan, index=data.index, dtype=float)

    in_position = False
    stop_price = np.nan

    for i in range(len(data.index)):
        if not in_position:
            if entry_conditions.iloc[i]:
                in_position = True
                entry_price = close.iloc[i]
                stop_price = entry_price - ATR_STOP_MULTIPLIER * atr14.iloc[i]
        else:
            exit_triggered = (
                ema8.iloc[i] < ema21.iloc[i]
                or close.iloc[i] < sma50.iloc[i]
                or close.iloc[i] < stop_price
            )
            if exit_triggered:
                in_position = False
                stop_price = np.nan

        signals.iloc[i] = 1 if in_position else 0
        stops.iloc[i] = stop_price if in_position else np.nan

    position_sizes = pd.Series(
        np.where(signals == 1, 1.0, 0.0), index=data.index, dtype=float
    )

    return signals, position_sizes, stops


def print_signal_summary(signals: pd.Series) -> None:
    """Print total active/flat day counts and a yearly activity breakdown."""
    total_active = int((signals == 1).sum())
    total_flat = int((signals == 0).sum())

    print("\nSignal Summary")
    print("=" * 30)
    print(f"Days with signal=1 (active): {total_active}")
    print(f"Days with signal=0 (flat):   {total_flat}")

    print("\nYearly breakdown:")
    for year, group in signals.groupby(signals.index.year):
        active = int((group == 1).sum())
        total = len(group)
        pct = active / total * 100 if total else 0.0
        print(f"  {year}: {active}/{total} days active ({pct:.1f}%)")


def main() -> None:
    """Fetch BTC data, run BT1, and print/plot the results."""
    data = fetch_instrument_data("BTC-USDC")

    signals, position_sizes, stops = build_signals(data)
    print_signal_summary(signals)

    engine = BacktestEngine(data, initial_capital=10000.0)
    results = engine.run(signals, position_sizes, stops)
    metrics = engine.compute_metrics(results)

    engine.print_metrics(metrics, label="BT1_Technical")
    engine.plot_results(results, label="BT1_Technical")


if __name__ == "__main__":
    main()
