"""BT2 — BT1's technical strategy gated by a daily macro filter.

Reuses BT1's entry/exit logic unchanged. On top of it, a macro score
(0-3) computed from QQQ, UUP (dollar proxy), and HY credit spreads
gates new entries and scales position size. Existing trades are never
force-exited on a macro downgrade — only the technical exit conditions
can close a position once it's open.
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
from backtest.data_fetcher import fetch_all
from backtest.engine import BacktestEngine
from backtest.indicators import adx, atr, ema, rsi, sma, volume_sma

MACRO_LOOKBACK_DAYS = 20


def compute_macro_scores(
    data: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Compute the three macro sub-scores and their sum.

    Each of the three conditions below scores 1 (permissive) or 0:
        qqq_score:    QQQ 20-day return > 0 (risk-on)
        dxy_score:    UUP 20-day return < 0 (dollar weakening)
        spread_score: HY spread 20-day change < 0 (tightening or stable)

    Returns:
        (qqq_score, dxy_score, spread_score, macro_score) Series aligned
        to `data.index`.
    """
    qqq_return_20d = data["qqq_close"].pct_change(MACRO_LOOKBACK_DAYS)
    uup_return_20d = data["uup_close"].pct_change(MACRO_LOOKBACK_DAYS)
    hy_spread_change_20d = data["hy_spread"].diff(MACRO_LOOKBACK_DAYS)

    qqq_score = (qqq_return_20d > 0).astype(int)
    dxy_score = (uup_return_20d < 0).astype(int)
    spread_score = (hy_spread_change_20d < 0).astype(int)

    macro_score = qqq_score + dxy_score + spread_score

    return qqq_score, dxy_score, spread_score, macro_score


def compute_macro_filter(data: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Compute the daily macro score (0-3) and the position size it implies.

    macro_score == 3 -> position_size 1.0 (full)
    macro_score == 2 -> position_size 0.5 (half)
    macro_score <= 1 -> position_size 0.0 (flat, no new entries)

    Returns:
        (macro_score, position_size) Series aligned to `data.index`.
    """
    _qqq_score, _dxy_score, _spread_score, macro_score = compute_macro_scores(data)

    position_size = pd.Series(0.0, index=data.index)
    position_size[macro_score == 3] = 1.0
    position_size[macro_score == 2] = 0.5

    return macro_score, position_size


def compute_entry_conditions(data: pd.DataFrame) -> pd.Series:
    """Compute BT1's technical entry conditions (unchanged in BT2).

    Returns:
        Boolean Series aligned to `data.index`.
    """
    close = data["close"]
    volume = data["volume"]

    ema8 = ema(close, 8)
    ema21 = ema(close, 21)
    sma50 = sma(close, 50)
    sma200 = sma(close, 200)
    adx14 = adx(data["high"], data["low"], close, 14)
    vol_sma20 = volume_sma(volume, 20)
    rsi14 = rsi(close, 14)

    return (
        (ema8 > ema21)
        & (close > sma50)
        & (close > sma200)
        & (adx14 > ADX_TREND_THRESHOLD)
        & (volume > vol_sma20)
        & (rsi14 < RSI_OVEREXTENDED_THRESHOLD)
    )


def print_macro_block_diagnostics(data: pd.DataFrame) -> None:
    """Print a breakdown of days where the macro filter blocked an entry.

    Identifies every day the technical entry conditions fired but the
    macro score (<= 1) forced position_size to 0, then reports the
    forward 20-day BTC return from each of those days. This tells us
    whether the filter is correctly avoiding bad setups or incorrectly
    sitting out good ones.
    """
    entry_conditions = compute_entry_conditions(data)
    qqq_score, dxy_score, spread_score, macro_score = compute_macro_scores(data)
    forward_return_20d = data["close"].shift(-MACRO_LOOKBACK_DAYS) / data["close"] - 1

    blocked = entry_conditions & (macro_score <= 1)
    blocked_dates = data.index[blocked]

    divider = "=" * 78
    print(f"\n{divider}")
    print(f"Macro Filter Diagnostics — {len(blocked_dates)} blocked entries")
    print(divider)

    if len(blocked_dates) == 0:
        print("No days where technicals fired but the macro filter blocked entry.")
        print(f"{divider}\n")
        return

    for date in blocked_dates:
        failed = []
        if qqq_score.loc[date] == 0:
            failed.append("qqq_score")
        if dxy_score.loc[date] == 0:
            failed.append("dxy_score")
        if spread_score.loc[date] == 0:
            failed.append("spread_score")

        fwd_return = forward_return_20d.loc[date]
        fwd_str = f"{fwd_return * 100:+.2f}%" if pd.notna(fwd_return) else "n/a (insufficient history)"

        print(
            f"{date.date()}  BTC close={data['close'].loc[date]:.2f}  "
            f"macro_score={macro_score.loc[date]}  "
            f"failed=[{', '.join(failed)}]  "
            f"fwd_20d_return={fwd_str}"
        )

    valid_fwd_returns = forward_return_20d.loc[blocked_dates].dropna()
    if len(valid_fwd_returns) > 0:
        avg_fwd_return = valid_fwd_returns.mean()
        win_rate = (valid_fwd_returns > 0).mean() * 100
        print("-" * 78)
        print(
            f"Avg forward 20d return on blocked days: {avg_fwd_return * 100:+.2f}%  "
            f"({len(valid_fwd_returns)} days with full lookahead, "
            f"{win_rate:.1f}% positive)"
        )
    print(f"{divider}\n")


def build_signals(data: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Generate the signal, position-size, and stop series for BT2.

    Entry/exit technical conditions are identical to BT1. A new entry is
    additionally gated on the macro score implying a non-zero position
    size (macro_score >= 2); if macro_score <= 1, no new entry is taken
    regardless of technicals, but a position already open is left to run
    until a technical exit condition fires. Position size is locked in
    at entry and does not change mid-trade if the macro score later
    changes.

    Returns:
        (signals, position_sizes, stops) Series aligned to `data.index`.
    """
    close = data["close"]

    ema8 = ema(close, 8)
    ema21 = ema(close, 21)
    sma50 = sma(close, 50)
    atr14 = atr(data["high"], data["low"], close, 14)

    entry_conditions = compute_entry_conditions(data)
    _macro_score, macro_position_size = compute_macro_filter(data)

    signals = pd.Series(0, index=data.index, dtype=int)
    position_sizes = pd.Series(0.0, index=data.index, dtype=float)
    stops = pd.Series(np.nan, index=data.index, dtype=float)

    in_position = False
    stop_price = np.nan
    position_size_at_entry = 0.0

    for i in range(len(data.index)):
        if not in_position:
            if entry_conditions.iloc[i] and macro_position_size.iloc[i] > 0:
                in_position = True
                entry_price = close.iloc[i]
                stop_price = entry_price - ATR_STOP_MULTIPLIER * atr14.iloc[i]
                position_size_at_entry = macro_position_size.iloc[i]
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

    return signals, position_sizes, stops


def print_comparison(bt1_metrics: dict, bt2_metrics: dict) -> None:
    """Print a side-by-side comparison of BT1 vs BT2 metrics."""
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
    total_width = label_width + 2 * col_width

    divider = "=" * total_width
    print(f"\n{divider}")
    print("BT1_Technical vs BT2_Macro — Comparison")
    print(divider)
    print(f"{'Metric':<{label_width}}{'BT1_Technical':>{col_width}}{'BT2_Macro':>{col_width}}")
    print("-" * total_width)
    for name, key, fmt in rows:
        bt1_value = fmt.format(bt1_metrics[key])
        bt2_value = fmt.format(bt2_metrics[key])
        print(f"{name:<{label_width}}{bt1_value:>{col_width}}{bt2_value:>{col_width}}")
    print(f"{divider}\n")


def main() -> None:
    """Fetch combined BTC+macro data, run BT1 and BT2, and compare them."""
    data = fetch_all()

    bt1_signals, bt1_position_sizes, bt1_stops = bt1_build_signals(data)
    bt2_signals, bt2_position_sizes, bt2_stops = build_signals(data)

    print_signal_summary(bt2_signals)
    print_macro_block_diagnostics(data)

    bt1_engine = BacktestEngine(data, initial_capital=10000.0)
    bt1_results = bt1_engine.run(bt1_signals, bt1_position_sizes, bt1_stops)
    bt1_metrics = bt1_engine.compute_metrics(bt1_results)

    bt2_engine = BacktestEngine(data, initial_capital=10000.0)
    bt2_results = bt2_engine.run(bt2_signals, bt2_position_sizes, bt2_stops)
    bt2_metrics = bt2_engine.compute_metrics(bt2_results)

    bt2_engine.print_metrics(bt2_metrics, label="BT2_Macro")
    bt2_engine.plot_results(bt2_results, label="BT2_Macro")

    print_comparison(bt1_metrics, bt2_metrics)


if __name__ == "__main__":
    main()
