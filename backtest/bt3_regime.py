"""BT3 — BT2's macro-gated technical strategy layered with a regime detector.

Adds a daily regime classification on top of BT2:
    Regime A — normal conditions. Position sizing follows BT2's macro
        score unchanged (0.0, 0.5, or 1.0).
    Regime B — a scheduled macro event (FOMC decision, CPI, or NFP
        release) falls within 1 calendar day of the current date.
        Position size is capped at 0.5 regardless of the macro score.
    Regime C — a volatility shock: ATR(14) has spiked to more than 2x
        its own 20-day average AND VIX jumped more than 20% on the same
        day. New entries are blocked and size is forced to 0.0 for the
        shock day plus the following 2-day cooling-off period.

As in BT2, a regime downgrade never force-exits an open position —
only the technical exit conditions (inherited unchanged from BT1) can
close a trade once it's open. Regime B/C only gate new entries and the
size committed to them.
"""

import numpy as np
import pandas as pd

from backtest.bt1_technical import (
    ADX_TREND_THRESHOLD,
    ATR_STOP_MULTIPLIER,
    RSI_OVEREXTENDED_THRESHOLD,
    build_signals as bt1_build_signals,
    print_signal_summary,
)
from backtest.bt2_macro import build_signals as bt2_build_signals, compute_macro_filter
from backtest.data_fetcher import fetch_all
from backtest.engine import BacktestEngine
from backtest.indicators import adx, atr, ema, rsi, sma, volume_sma

# FOMC decision dates (second/announcement day of each two-day meeting).
# Source: federalreserve.gov/monetarypolicy/fomccalendars.htm — published
# years in advance, so the 2025 dates were known well before those meetings.
FOMC_DATES = [
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
]

# CPI (Consumer Price Index) release dates. Source: BLS "Schedule of
# Releases for the Consumer Price Index". The October 2025 release
# (covering October data) was cancelled outright due to the 2025 federal
# government shutdown; the November-data release slipped from its
# originally scheduled Dec 10 to Dec 18.
CPI_DATES = [
    "2023-01-12", "2023-02-14", "2023-03-14", "2023-04-12",
    "2023-05-10", "2023-06-13", "2023-07-12", "2023-08-10",
    "2023-09-13", "2023-10-12", "2023-11-14", "2023-12-12",
    "2024-01-11", "2024-02-13", "2024-03-12", "2024-04-10",
    "2024-05-15", "2024-06-12", "2024-07-11", "2024-08-14",
    "2024-09-11", "2024-10-10", "2024-11-13", "2024-12-11",
    "2025-01-15", "2025-02-12", "2025-03-12", "2025-04-10",
    "2025-05-13", "2025-06-11", "2025-07-15", "2025-08-12",
    "2025-09-11", "2025-10-24", "2025-12-18",
]

# NFP (Employment Situation / nonfarm payrolls) release dates. Source: BLS
# "Schedule of Releases for the Employment Situation" — normally the first
# Friday of the month. The September-2025 report was delayed to Nov 20,
# 2025 and the October-2025 report was folded into a combined
# October+November release on Dec 16, 2025, both due to the 2025 federal
# government shutdown.
NFP_DATES = [
    "2023-01-06", "2023-02-03", "2023-03-10", "2023-04-07",
    "2023-05-05", "2023-06-02", "2023-07-07", "2023-08-04",
    "2023-09-01", "2023-10-06", "2023-11-03", "2023-12-08",
    "2024-01-05", "2024-02-02", "2024-03-08", "2024-04-05",
    "2024-05-03", "2024-06-07", "2024-07-05", "2024-08-02",
    "2024-09-06", "2024-10-04", "2024-11-01", "2024-12-06",
    "2025-01-10", "2025-02-07", "2025-03-07", "2025-04-04",
    "2025-05-02", "2025-06-06", "2025-07-03", "2025-08-01",
    "2025-09-05", "2025-11-20", "2025-12-16",
]

REGIME_B_EVENT_WINDOW_DAYS = 1
REGIME_B_MAX_POSITION_SIZE = 0.5

ATR_SPIKE_SMA_PERIOD = 20
ATR_SPIKE_MULTIPLIER = 2.0
VIX_SHOCK_PCT = 0.20
REGIME_C_COOLDOWN_DAYS = 2


def compute_regime_b(index: pd.DatetimeIndex) -> pd.Series:
    """Flag each date in `index` that falls within 1 day of a scheduled event.

    Combines FOMC, CPI, and NFP dates into one event set and marks the day
    before, day of, and day after each event.

    Returns:
        Boolean Series aligned to `index`.
    """
    event_dates = pd.to_datetime(sorted(set(FOMC_DATES) | set(CPI_DATES) | set(NFP_DATES)))

    window_days = set()
    for event_date in event_dates:
        for offset in range(-REGIME_B_EVENT_WINDOW_DAYS, REGIME_B_EVENT_WINDOW_DAYS + 1):
            window_days.add((event_date + pd.Timedelta(days=offset)).date())

    return pd.Series([ts.date() in window_days for ts in index], index=index)


def compute_regime_c(data: pd.DataFrame, atr14: pd.Series) -> pd.Series:
    """Flag volatility-shock days plus their 2-day cooling-off period.

    A shock day requires both, on the same day:
        - ATR(14) > 2.0x its own 20-day SMA (volatility spiking)
        - VIX day-over-day change > 20% (VIX jumping sharply)

    Once a shock day fires, that day and the following
    `REGIME_C_COOLDOWN_DAYS` days are also flagged.

    Returns:
        Boolean Series aligned to `data.index`.
    """
    atr_sma20 = sma(atr14, ATR_SPIKE_SMA_PERIOD)
    volatility_spike = atr14 > ATR_SPIKE_MULTIPLIER * atr_sma20

    vix_change = data["vix_close"].pct_change()
    vix_shock = vix_change > VIX_SHOCK_PCT

    shock_day = (volatility_spike & vix_shock).fillna(False)

    regime_c = shock_day.copy()
    n = len(regime_c)
    for i in np.where(shock_day.to_numpy())[0]:
        for offset in range(1, REGIME_C_COOLDOWN_DAYS + 1):
            if i + offset < n:
                regime_c.iloc[i + offset] = True

    return regime_c


def build_signals(data: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Generate the signal, position-size, stop, and regime series for BT3.

    Entry/exit technical conditions and the macro score are identical to
    BT1/BT2. On top of that, a new entry is gated by the day's regime:
        Regime C: no new entries; size forced to 0.0.
        Regime B: entries allowed under BT2's macro gate, but size is
            capped at REGIME_B_MAX_POSITION_SIZE.
        Regime A: BT2's macro-derived size applies unchanged.
    A position already open is never force-exited on a regime change —
    only the technical exit conditions can close it, and its size stays
    locked in at whatever was committed on entry.

    Returns:
        (signals, position_sizes, stops, regime) Series aligned to
        `data.index`. `regime` holds 'A', 'B', or 'C' per day.
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

    _macro_score, macro_position_size = compute_macro_filter(data)
    regime_b = compute_regime_b(data.index)
    regime_c = compute_regime_c(data, atr14)

    regime = pd.Series("A", index=data.index, dtype=object)
    regime[regime_b] = "B"
    regime[regime_c] = "C"

    signals = pd.Series(0, index=data.index, dtype=int)
    position_sizes = pd.Series(0.0, index=data.index, dtype=float)
    stops = pd.Series(np.nan, index=data.index, dtype=float)

    in_position = False
    stop_price = np.nan
    position_size_at_entry = 0.0

    for i in range(len(data.index)):
        if not in_position:
            if regime_c.iloc[i]:
                desired_size = 0.0
            elif regime_b.iloc[i]:
                desired_size = min(macro_position_size.iloc[i], REGIME_B_MAX_POSITION_SIZE)
            else:
                desired_size = macro_position_size.iloc[i]

            if entry_conditions.iloc[i] and desired_size > 0:
                in_position = True
                entry_price = close.iloc[i]
                stop_price = entry_price - ATR_STOP_MULTIPLIER * atr14.iloc[i]
                position_size_at_entry = desired_size
        else:
            exit_triggered = (
                ema8.iloc[i] < ema21.iloc[i]
                or close.iloc[i] < sma50.iloc[i]
                or close.iloc[i] < stop_price
            )
            if exit_triggered:
                in_position = False
                stop_price = np.nan
                position_size_at_entry = 0.0

        signals.iloc[i] = 1 if in_position else 0
        stops.iloc[i] = stop_price if in_position else np.nan
        position_sizes.iloc[i] = position_size_at_entry if in_position else 0.0

    return signals, position_sizes, stops, regime


def print_regime_summary(regime: pd.Series) -> None:
    """Print how many days (and what %) were spent in each regime."""
    total = len(regime)
    counts = regime.value_counts()

    print("\nRegime Summary")
    print("=" * 30)
    for label in ("A", "B", "C"):
        count = int(counts.get(label, 0))
        pct = count / total * 100 if total else 0.0
        print(f"  Regime {label}: {count}/{total} days ({pct:.1f}%)")
    print()


def print_comparison(bt1_metrics: dict, bt2_metrics: dict, bt3_metrics: dict) -> None:
    """Print a three-way side-by-side comparison of BT1 vs BT2 vs BT3 metrics."""
    rows = [
        ("Total Return", "total_return_pct", "{:.2f}%"),
        ("BTC Buy & Hold Return", "btc_hold_return_pct", "{:.2f}%"),
        ("Sharpe Ratio", "sharpe_ratio", "{:.2f}"),
        ("Sortino Ratio", "sortino_ratio", "{:.2f}"),
        ("Max Drawdown", "max_drawdown_pct", "{:.2f}%"),
        ("Max Drawdown Duration", "max_drawdown_duration_days", "{} days"),
        ("Calmar Ratio", "calmar_ratio", "{:.2f}"),
        ("Win Rate", "win_rate_pct", "{:.2f}%"),
        ("Avg Trade Duration", "avg_trade_duration_days", "{:.1f} days"),
        ("Num Trades", "num_trades", "{}"),
    ]

    label_width = max(len(name) for name, _, _ in rows) + 2
    col_width = 18
    total_width = label_width + 3 * col_width

    divider = "=" * total_width
    print(f"\n{divider}")
    print("BT1_Technical vs BT2_Macro vs BT3_Regime — Comparison")
    print(divider)
    print(
        f"{'Metric':<{label_width}}"
        f"{'BT1_Technical':>{col_width}}"
        f"{'BT2_Macro':>{col_width}}"
        f"{'BT3_Regime':>{col_width}}"
    )
    print("-" * total_width)
    for name, key, fmt in rows:
        bt1_value = fmt.format(bt1_metrics[key])
        bt2_value = fmt.format(bt2_metrics[key])
        bt3_value = fmt.format(bt3_metrics[key])
        print(f"{name:<{label_width}}{bt1_value:>{col_width}}{bt2_value:>{col_width}}{bt3_value:>{col_width}}")
    print(f"{divider}\n")


def main() -> None:
    """Fetch combined BTC+macro data, run BT1, BT2, and BT3, and compare them."""
    data = fetch_all()

    bt1_signals, bt1_position_sizes, bt1_stops = bt1_build_signals(data)
    bt2_signals, bt2_position_sizes, bt2_stops = bt2_build_signals(data)
    bt3_signals, bt3_position_sizes, bt3_stops, regime = build_signals(data)

    print_signal_summary(bt3_signals)
    print_regime_summary(regime)

    bt1_engine = BacktestEngine(data, initial_capital=10000.0)
    bt1_results = bt1_engine.run(bt1_signals, bt1_position_sizes, bt1_stops)
    bt1_metrics = bt1_engine.compute_metrics(bt1_results)

    bt2_engine = BacktestEngine(data, initial_capital=10000.0)
    bt2_results = bt2_engine.run(bt2_signals, bt2_position_sizes, bt2_stops)
    bt2_metrics = bt2_engine.compute_metrics(bt2_results)

    bt3_engine = BacktestEngine(data, initial_capital=10000.0)
    bt3_results = bt3_engine.run(bt3_signals, bt3_position_sizes, bt3_stops)
    bt3_results["regime"] = regime.to_numpy()
    bt3_metrics = bt3_engine.compute_metrics(bt3_results)

    bt3_engine.print_metrics(bt3_metrics, label="BT3_Regime")
    bt3_engine.plot_results(bt3_results, label="BT3_Regime")

    print_comparison(bt1_metrics, bt2_metrics, bt3_metrics)


if __name__ == "__main__":
    main()
