"""BT4 — BT2's macro-gated technical strategy, refined in three ways.

Starts from BT2 (technical entries/exits gated by a macro filter) and
makes exactly three changes:

1. Weighted macro scoring: the three macro sub-scores (QQQ, dollar, HY
   spread) are combined with weights 0.45/0.20/0.35 instead of counted
   equally, with new thresholds (0.60 full, 0.35 half) on the resulting
   0-1 score.
2. BTC momentum override: a strong BTC 20-day return can push a red or
   neutral macro multiplier up one notch (0.0 -> 0.5, or 0.5 -> 0.75),
   on the theory that powerful price momentum can outweigh a
   lukewarm/negative macro backdrop.
3. Continuous technical conviction: BT1's six binary entry conditions
   are scored 1/0 and averaged into a technical_confidence in [0, 1]
   instead of requiring all six to hold. Final position size is
   technical_confidence x macro_multiplier, floored to flat below 0.5
   to avoid fragmented micro-positions.

Exit logic is inherited unchanged from BT1/BT2: EMA(8) crossing below
EMA(21), close falling below SMA(50), or a breach of the ATR stop set
at entry. As in BT2/BT3, a position already open is never force-exited
on a macro/momentum change — only those technical exit conditions can
close a trade, and its size stays locked in at whatever was committed
on entry.
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
from backtest.bt2_macro import build_signals as bt2_build_signals, compute_macro_scores
from backtest.bt3_regime import build_signals as bt3_build_signals
from backtest.data_fetcher import fetch_all
from backtest.engine import BacktestEngine
from backtest.indicators import adx, atr, ema, rsi, sma, volume_sma

QQQ_WEIGHT = 0.45
DXY_WEIGHT = 0.20
SPREAD_WEIGHT = 0.35

MACRO_FULL_THRESHOLD = 0.60
MACRO_HALF_THRESHOLD = 0.35

BTC_MOMENTUM_LOOKBACK_DAYS = 20
BTC_OVERRIDE_RED_THRESHOLD = 0.15
BTC_OVERRIDE_NEUTRAL_THRESHOLD = 0.25
OVERRIDE_TO_HALF = 0.5
OVERRIDE_TO_THREE_QUARTER = 0.75

NUM_TECHNICAL_CONDITIONS = 6
MIN_POSITION_SIZE_THRESHOLD = 0.5


def compute_weighted_macro_filter(data: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Compute the weighted macro score and the base position-size multiplier.

    weighted_macro_score = qqq_score*0.45 + dxy_score*0.20 + spread_score*0.35,
    using the same binary component scores as BT2.

    weighted_macro_score >= 0.60 -> multiplier 1.0 (full)
    weighted_macro_score >= 0.35 -> multiplier 0.5 (half)
    weighted_macro_score <  0.35 -> multiplier 0.0 (flat)

    Returns:
        (weighted_macro_score, base_multiplier) Series aligned to `data.index`.
    """
    qqq_score, dxy_score, spread_score, _macro_score = compute_macro_scores(data)

    weighted_macro_score = (
        qqq_score * QQQ_WEIGHT + dxy_score * DXY_WEIGHT + spread_score * SPREAD_WEIGHT
    )

    base_multiplier = pd.Series(0.0, index=data.index)
    base_multiplier[weighted_macro_score >= MACRO_FULL_THRESHOLD] = 1.0
    base_multiplier[
        (weighted_macro_score >= MACRO_HALF_THRESHOLD)
        & (weighted_macro_score < MACRO_FULL_THRESHOLD)
    ] = 0.5

    return weighted_macro_score, base_multiplier


def compute_btc_20d_return(data: pd.DataFrame) -> pd.Series:
    """BTC's trailing 20-day return: (close - close.shift(20)) / close.shift(20)."""
    close = data["close"]
    return (close - close.shift(BTC_MOMENTUM_LOOKBACK_DAYS)) / close.shift(
        BTC_MOMENTUM_LOOKBACK_DAYS
    )


def apply_momentum_override(
    base_multiplier: pd.Series, btc_20d_return: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """Bump a red/neutral macro multiplier up one notch on strong BTC momentum.

    If the base multiplier is 0.0 (red) and btc_20d_return > 0.15, override
    to 0.5. If the base multiplier is 0.5 (neutral) and btc_20d_return >
    0.25, override to 0.75. A full (1.0) multiplier is never touched.

    Returns:
        (final_multiplier, override_fired) Series aligned to `data.index`.
        `override_fired` is True on any day either override condition hit.
    """
    red_override = (base_multiplier == 0.0) & (btc_20d_return > BTC_OVERRIDE_RED_THRESHOLD)
    neutral_override = (base_multiplier == 0.5) & (
        btc_20d_return > BTC_OVERRIDE_NEUTRAL_THRESHOLD
    )

    final_multiplier = base_multiplier.copy()
    final_multiplier[red_override] = OVERRIDE_TO_HALF
    final_multiplier[neutral_override] = OVERRIDE_TO_THREE_QUARTER

    override_fired = (red_override | neutral_override).fillna(False)

    return final_multiplier, override_fired


def compute_macro_filter(
    data: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Compute the full BT4 macro pipeline: weighted score, momentum override, final multiplier.

    Returns:
        (weighted_macro_score, btc_20d_return, macro_multiplier, override_fired)
        Series aligned to `data.index`.
    """
    weighted_macro_score, base_multiplier = compute_weighted_macro_filter(data)
    btc_20d_return = compute_btc_20d_return(data)
    macro_multiplier, override_fired = apply_momentum_override(base_multiplier, btc_20d_return)

    return weighted_macro_score, btc_20d_return, macro_multiplier, override_fired


def compute_technical_confidence(data: pd.DataFrame) -> pd.Series:
    """Score BT1's six entry conditions 1/0 and average them into [0, 1].

    The six conditions (identical to BT1's binary entry gate) are:
    EMA(8) > EMA(21), close > SMA(50), close > SMA(200), ADX(14) > 25,
    volume > its 20-day average, and RSI(14) < 75.

    Returns:
        Float Series aligned to `data.index`.
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

    condition_sum = (
        (ema8 > ema21).astype(int)
        + (close > sma50).astype(int)
        + (close > sma200).astype(int)
        + (adx14 > ADX_TREND_THRESHOLD).astype(int)
        + (volume > vol_sma20).astype(int)
        + (rsi14 < RSI_OVEREXTENDED_THRESHOLD).astype(int)
    )

    return condition_sum / NUM_TECHNICAL_CONDITIONS


def print_momentum_override_log(
    data: pd.DataFrame, btc_20d_return: pd.Series, override_fired: pd.Series
) -> None:
    """Print every date the BTC momentum override fired, with fwd 20d return.

    For each override event, reports the BTC 20-day return that triggered
    it and the forward 20-day return from that date, so the override's
    real-world value can be sanity-checked.
    """
    forward_return_20d = (
        data["close"].shift(-BTC_MOMENTUM_LOOKBACK_DAYS) / data["close"] - 1
    )
    override_dates = data.index[override_fired]

    divider = "=" * 78
    print(f"\n{divider}")
    print(f"BTC Momentum Override Log — {len(override_dates)} override events")
    print(divider)

    if len(override_dates) == 0:
        print("No momentum override events fired.")
        print(f"{divider}\n")
        return

    for date in override_dates:
        fwd_return = forward_return_20d.loc[date]
        fwd_str = f"{fwd_return * 100:+.2f}%" if pd.notna(fwd_return) else "n/a (insufficient history)"
        print(
            f"{date.date()}  BTC close={data['close'].loc[date]:.2f}  "
            f"btc_20d_return={btc_20d_return.loc[date] * 100:+.2f}%  "
            f"fwd_20d_return={fwd_str}"
        )

    valid_fwd_returns = forward_return_20d.loc[override_dates].dropna()
    if len(valid_fwd_returns) > 0:
        avg_fwd_return = valid_fwd_returns.mean()
        win_rate = (valid_fwd_returns > 0).mean() * 100
        print("-" * 78)
        print(
            f"Avg forward 20d return on override days: {avg_fwd_return * 100:+.2f}%  "
            f"({len(valid_fwd_returns)} days with full lookahead, "
            f"{win_rate:.1f}% positive)"
        )
    print(f"{divider}\n")


def build_signals(
    data: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Generate the signal, position-size, and stop series for BT4.

    Exit logic is identical to BT1/BT2. Entry replaces BT2's all-or-nothing
    technical gate: each day's candidate position size is
    technical_confidence x macro_multiplier, and a new entry is taken only
    once that candidate size reaches MIN_POSITION_SIZE_THRESHOLD (0.5).
    As in BT2/BT3, position size is locked in at entry and does not change
    mid-trade if the underlying scores later change.

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

    for i in range(len(data.index)):
        if not in_position:
            if candidate_position_size.iloc[i] >= MIN_POSITION_SIZE_THRESHOLD:
                in_position = True
                entry_price = close.iloc[i]
                stop_price = entry_price - ATR_STOP_MULTIPLIER * atr14.iloc[i]
                position_size_at_entry = candidate_position_size.iloc[i]
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
    bt1_metrics: dict, bt2_metrics: dict, bt3_metrics: dict, bt4_metrics: dict
) -> None:
    """Print a four-way side-by-side comparison of BT1 vs BT2 vs BT3 vs BT4 metrics."""
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
    col_width = 16
    total_width = label_width + 4 * col_width

    divider = "=" * total_width
    print(f"\n{divider}")
    print("BT1_Technical vs BT2_Macro vs BT3_Regime vs BT4_Refined — Comparison")
    print(divider)
    print(
        f"{'Metric':<{label_width}}"
        f"{'BT1_Technical':>{col_width}}"
        f"{'BT2_Macro':>{col_width}}"
        f"{'BT3_Regime':>{col_width}}"
        f"{'BT4_Refined':>{col_width}}"
    )
    print("-" * total_width)
    for name, key, fmt in rows:
        bt1_value = fmt.format(bt1_metrics[key])
        bt2_value = fmt.format(bt2_metrics[key])
        bt3_value = fmt.format(bt3_metrics[key])
        bt4_value = fmt.format(bt4_metrics[key])
        print(
            f"{name:<{label_width}}"
            f"{bt1_value:>{col_width}}"
            f"{bt2_value:>{col_width}}"
            f"{bt3_value:>{col_width}}"
            f"{bt4_value:>{col_width}}"
        )
    print(f"{divider}\n")


def print_commission_comparison(no_commission_metrics: dict, with_commission_metrics: dict) -> None:
    """Print BT4 metrics with and without commissions side by side."""
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
    print("BT4_Refined — Commission Impact (0.1% Coinbase taker fee)")
    print(divider)
    print(
        f"{'Metric':<{label_width}}"
        f"{'No Commission':>{col_width}}"
        f"{'With Commission':>{col_width}}"
    )
    print("-" * total_width)
    for name, key, fmt in rows:
        no_commission_value = fmt.format(no_commission_metrics[key])
        with_commission_value = fmt.format(with_commission_metrics[key])
        print(
            f"{name:<{label_width}}"
            f"{no_commission_value:>{col_width}}"
            f"{with_commission_value:>{col_width}}"
        )
    print(f"{divider}\n")


def print_trade_log(trades: list[dict]) -> None:
    """Print every individual BT4 trade, then avg return per trade by year.

    Args:
        trades: BacktestEngine.trades from a completed run() call, each
            entry with entry/exit date & price, return_pct, duration_days,
            and exit_reason ("technical_exit", "stop_hit", or "end_of_data").
    """
    divider = "=" * 96
    print(f"\n{divider}")
    print(f"BT4_Refined — Trade Log ({len(trades)} trades)")
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
    print("BT4_Refined — Avg Return per Trade by Year")
    print(divider)
    print(f"{'Year':<8}{'Num Trades':>12}{'Avg Return %':>16}")
    print("-" * 46)
    for year in sorted(returns_by_year):
        year_returns = returns_by_year[year]
        avg_return_pct = sum(year_returns) / len(year_returns)
        print(f"{year:<8}{len(year_returns):>12}{avg_return_pct:>+15.2f}%")
    print(f"{divider}\n")


def main() -> None:
    """Fetch combined BTC+macro data, run BT1-BT4, and compare them."""
    data = fetch_all()

    bt1_signals, bt1_position_sizes, bt1_stops = bt1_build_signals(data)
    bt2_signals, bt2_position_sizes, bt2_stops = bt2_build_signals(data)
    bt3_signals, bt3_position_sizes, bt3_stops, _regime = bt3_build_signals(data)
    (
        bt4_signals,
        bt4_position_sizes,
        bt4_stops,
        _technical_confidence,
        _macro_multiplier,
        override_fired,
        btc_20d_return,
    ) = build_signals(data)

    print_signal_summary(bt4_signals)
    print_momentum_override_log(data, btc_20d_return, override_fired)

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

    bt4_engine.print_metrics(bt4_metrics, label="BT4_Refined")
    bt4_engine.plot_results(bt4_results, label="BT4_Refined")

    print_comparison(bt1_metrics, bt2_metrics, bt3_metrics, bt4_metrics)

    bt4_no_commission_engine = BacktestEngine(data, initial_capital=10000.0, commission_rate=0.0)
    bt4_no_commission_results = bt4_no_commission_engine.run(
        bt4_signals, bt4_position_sizes, bt4_stops
    )
    bt4_no_commission_metrics = bt4_no_commission_engine.compute_metrics(
        bt4_no_commission_results
    )

    print_commission_comparison(bt4_no_commission_metrics, bt4_metrics)

    print_trade_log(bt4_engine.trades)


if __name__ == "__main__":
    main()
