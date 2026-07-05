"""BT5 — BT4's refined strategy, filtered in two ways.

Starts from BT4 (weighted macro scoring + BTC momentum override +
continuous technical conviction) and makes exactly two changes:

1. Minimum hold period: once a trade is entered, technical exit
   conditions (EMA(8) crossing below EMA(21), close falling below
   SMA(50)) are suppressed for the first `min_hold_days` (default 5)
   days of the trade. The ATR-based stop (close breaching entry_price -
   2xATR(14)) remains active from day 1 regardless of hold period —
   only the two technical exits are gated by the hold window.
2. Raised conviction threshold: a new entry requires candidate position
   size (technical_confidence x macro_multiplier) >= 0.65, up from
   BT4's 0.5. This higher bar applies only at entry — once in a trade,
   exits are governed solely by the technical/stop conditions above,
   never by the entry threshold.

Everything else — weighted macro scoring, the BTC momentum override,
continuous technical conviction scoring, and 0.1% commission modeling —
is inherited unchanged from BT4.
"""

import numpy as np
import pandas as pd

from backtest.bt1_technical import ATR_STOP_MULTIPLIER, build_signals as bt1_build_signals
from backtest.bt2_macro import build_signals as bt2_build_signals
from backtest.bt3_regime import build_signals as bt3_build_signals
from backtest.bt4_refined import (
    build_signals as bt4_build_signals,
    compute_macro_filter,
    compute_technical_confidence,
)
from backtest.data_fetcher import fetch_all
from backtest.engine import BacktestEngine
from backtest.indicators import atr, ema, sma

MIN_HOLD_DAYS = 5
MIN_POSITION_SIZE_THRESHOLD = 0.65


def build_signals(
    data: pd.DataFrame, min_hold_days: int = MIN_HOLD_DAYS
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Generate the signal, position-size, and stop series for BT5.

    Identical to BT4 (bt4_refined.build_signals) except for the two
    changes described in the module docstring: a minimum hold period
    that suppresses technical exits early in a trade, and a raised
    0.65 entry threshold on candidate position size.

    Returns:
        (signals, position_sizes, stops, technical_confidence,
        macro_multiplier, override_fired, btc_20d_return) Series aligned to
        `data.index`.
    """
    close = data["close"]

    ema8 = ema(close, 8)
    ema21 = ema(close, 21)
    sma50 = sma(close, 50)
    atr14 = atr(data["high"], data["low"], close, 14)

    technical_confidence = compute_technical_confidence(data)
    _weighted_macro_score, btc_20d_return, macro_multiplier, override_fired = compute_macro_filter(
        data
    )

    candidate_position_size = technical_confidence * macro_multiplier

    signals = pd.Series(0, index=data.index, dtype=int)
    position_sizes = pd.Series(0.0, index=data.index, dtype=float)
    stops = pd.Series(np.nan, index=data.index, dtype=float)

    in_position = False
    stop_price = np.nan
    position_size_at_entry = 0.0
    days_in_trade = 0

    for i in range(len(data.index)):
        if not in_position:
            if candidate_position_size.iloc[i] >= MIN_POSITION_SIZE_THRESHOLD:
                in_position = True
                entry_price = close.iloc[i]
                stop_price = entry_price - ATR_STOP_MULTIPLIER * atr14.iloc[i]
                position_size_at_entry = candidate_position_size.iloc[i]
                days_in_trade = 0
        else:
            days_in_trade += 1

            stop_triggered = close.iloc[i] < stop_price
            technical_exit_triggered = (
                ema8.iloc[i] < ema21.iloc[i] or close.iloc[i] < sma50.iloc[i]
            )
            exit_triggered = stop_triggered or (
                technical_exit_triggered and days_in_trade >= min_hold_days
            )

            if exit_triggered:
                in_position = False
                stop_price = np.nan
                position_size_at_entry = 0.0
                days_in_trade = 0

        signals.iloc[i] = 1 if in_position else 0
        stops.iloc[i] = stop_price if in_position else np.nan
        position_sizes.iloc[i] = position_size_at_entry if in_position else 0.0

    return (
        signals,
        position_sizes,
        stops,
        technical_confidence,
        macro_multiplier,
        override_fired,
        btc_20d_return,
    )


def print_comparison(
    bt1_metrics: dict,
    bt2_metrics: dict,
    bt3_metrics: dict,
    bt4_metrics: dict,
    bt5_metrics: dict,
) -> None:
    """Print a five-way side-by-side comparison of BT1-BT5 metrics."""
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
        ("Pct Return Top 3 Trades", "pct_return_from_top_3_trades", "{:.2f}%"),
    ]

    label_width = max(len(name) for name, _, _ in rows) + 2
    col_width = 14
    total_width = label_width + 5 * col_width

    divider = "=" * total_width
    print(f"\n{divider}")
    print("BT1_Technical vs BT2_Macro vs BT3_Regime vs BT4_Refined vs BT5_Filtered — Comparison")
    print(divider)
    print(
        f"{'Metric':<{label_width}}"
        f"{'BT1':>{col_width}}"
        f"{'BT2':>{col_width}}"
        f"{'BT3':>{col_width}}"
        f"{'BT4':>{col_width}}"
        f"{'BT5':>{col_width}}"
    )
    print("-" * total_width)
    for name, key, fmt in rows:
        bt1_value = fmt.format(bt1_metrics[key])
        bt2_value = fmt.format(bt2_metrics[key])
        bt3_value = fmt.format(bt3_metrics[key])
        bt4_value = fmt.format(bt4_metrics[key])
        bt5_value = fmt.format(bt5_metrics[key])
        print(
            f"{name:<{label_width}}"
            f"{bt1_value:>{col_width}}"
            f"{bt2_value:>{col_width}}"
            f"{bt3_value:>{col_width}}"
            f"{bt4_value:>{col_width}}"
            f"{bt5_value:>{col_width}}"
        )
    print(f"{divider}\n")


def print_trade_log(trades: list[dict]) -> None:
    """Print every individual BT5 trade, then avg return per trade by year.

    Args:
        trades: BacktestEngine.trades from a completed run() call, each
            entry with entry/exit date & price, return_pct, duration_days,
            and exit_reason ("technical_exit", "stop_hit", or "end_of_data").
    """
    divider = "=" * 96
    print(f"\n{divider}")
    print(f"BT5_Filtered — Trade Log ({len(trades)} trades)")
    print(divider)
    print(
        f"{'Entry Date':<12}{'Entry Price':>13}  {'Exit Date':<12}"
        f"{'Exit Price':>13}{'Duration':>11}{'Return %':>11}  {'Exit Reason'}"
    )
    print("-" * 96)
    for trade in trades:
        print(
            f"{str(trade['entry_date'].date()):<12}"
            f"{trade['entry_price']:>13,.2f}  "
            f"{str(trade['exit_date'].date()):<12}"
            f"{trade['exit_price']:>13,.2f}"
            f"{trade['duration_days']:>9}d "
            f"{trade['return_pct']:>+10.2f}%  "
            f"{trade['exit_reason']}"
        )
    print(f"{divider}\n")

    returns_by_year: dict[int, list[float]] = {}
    for trade in trades:
        returns_by_year.setdefault(trade["entry_date"].year, []).append(trade["return_pct"])

    divider = "=" * 46
    print(f"\n{divider}")
    print("BT5_Filtered — Avg Return per Trade by Year")
    print(divider)
    print(f"{'Year':<8}{'Num Trades':>12}{'Avg Return %':>16}")
    print("-" * 46)
    for year in sorted(returns_by_year):
        year_returns = returns_by_year[year]
        avg_return_pct = sum(year_returns) / len(year_returns)
        print(f"{year:<8}{len(year_returns):>12}{avg_return_pct:>+15.2f}%")
    print(f"{divider}\n")


def main() -> None:
    """Fetch combined BTC+macro data, run BT1-BT5, and compare them."""
    data = fetch_all()

    bt1_signals, bt1_position_sizes, bt1_stops = bt1_build_signals(data)
    bt2_signals, bt2_position_sizes, bt2_stops = bt2_build_signals(data)
    bt3_signals, bt3_position_sizes, bt3_stops, _regime = bt3_build_signals(data)
    (
        bt4_signals,
        bt4_position_sizes,
        bt4_stops,
        _bt4_technical_confidence,
        _bt4_macro_multiplier,
        _bt4_override_fired,
        _bt4_btc_20d_return,
    ) = bt4_build_signals(data)
    (
        bt5_signals,
        bt5_position_sizes,
        bt5_stops,
        _technical_confidence,
        _macro_multiplier,
        _override_fired,
        _btc_20d_return,
    ) = build_signals(data)

    bt1_engine = BacktestEngine(data, initial_capital=10000.0)
    bt1_results = bt1_engine.run(bt1_signals, bt1_position_sizes, bt1_stops)
    bt1_metrics = bt1_engine.compute_metrics(bt1_results)

    bt2_engine = BacktestEngine(data, initial_capital=10000.0)
    bt2_results = bt2_engine.run(bt2_signals, bt2_position_sizes, bt2_stops)
    bt2_metrics = bt2_engine.compute_metrics(bt2_results)

    bt3_engine = BacktestEngine(data, initial_capital=10000.0)
    bt3_results = bt3_engine.run(bt3_signals, bt3_position_sizes, bt3_stops)
    bt3_metrics = bt3_engine.compute_metrics(bt3_results)

    bt4_engine = BacktestEngine(data, initial_capital=10000.0)
    bt4_results = bt4_engine.run(bt4_signals, bt4_position_sizes, bt4_stops)
    bt4_metrics = bt4_engine.compute_metrics(bt4_results)

    bt5_engine = BacktestEngine(data, initial_capital=10000.0)
    bt5_results = bt5_engine.run(bt5_signals, bt5_position_sizes, bt5_stops)
    bt5_metrics = bt5_engine.compute_metrics(bt5_results)

    bt5_engine.print_metrics(bt5_metrics, label="BT5_Filtered")
    bt5_engine.plot_results(bt5_results, label="BT5_Filtered")

    print_trade_log(bt5_engine.trades)

    print_comparison(bt1_metrics, bt2_metrics, bt3_metrics, bt4_metrics, bt5_metrics)


if __name__ == "__main__":
    main()
