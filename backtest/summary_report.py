"""summary_report.py — runs BT1, BT2, and BT3 back to back on the same
data and produces a side-by-side comparison: a console metrics table
with a per-metric winner, a three-panel comparison chart, and a
plain-English readout of what the numbers mean.

Reuses each backtest's `build_signals()` directly (rather than calling
their `main()` functions) so all three run against one shared
`fetch_all()` pull and one `BacktestEngine` per strategy.
"""

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from backtest.bt1_technical import build_signals as bt1_build_signals
from backtest.bt2_macro import build_signals as bt2_build_signals
from backtest.bt3_regime import build_signals as bt3_build_signals
from backtest.data_fetcher import fetch_all
from backtest.engine import RESULTS_DIR, TRADING_DAYS_PER_YEAR, BacktestEngine

INITIAL_CAPITAL = 10000.0
ROLLING_SHARPE_WINDOW_DAYS = 90

STRATEGY_LABELS = ("BT1_Technical", "BT2_Macro", "BT3_Regime")
STRATEGY_COLORS = {
    "BT1_Technical": "#2a78d6",
    "BT2_Macro": "#3fa34d",
    "BT3_Regime": "#c0392b",
}
BTC_HOLD_COLOR = "#eb6834"

# (metrics key, display name, format string, "higher wins"/"lower wins"/None)
# None means the metric has no inherent winner (it's informational only).
METRICS = [
    ("total_return_pct", "Total Return", "{:.2f}%", True),
    ("btc_hold_return_pct", "BTC Buy & Hold Return", "{:.2f}%", None),
    ("sharpe_ratio", "Sharpe Ratio", "{:.2f}", True),
    ("sortino_ratio", "Sortino Ratio", "{:.2f}", True),
    ("max_drawdown_pct", "Max Drawdown", "{:.2f}%", True),
    ("max_drawdown_duration_days", "Max Drawdown Duration", "{} days", False),
    ("calmar_ratio", "Calmar Ratio", "{:.2f}", True),
    ("win_rate_pct", "Win Rate", "{:.2f}%", True),
    ("avg_trade_duration_days", "Avg Trade Duration", "{:.1f} days", None),
    ("num_trades", "Num Trades", "{}", None),
]


def run_all_backtests() -> dict[str, dict]:
    """Fetch one shared dataset and run BT1, BT2, and BT3 against it.

    Returns:
        Dict keyed by strategy label, each holding {"results": DataFrame,
        "metrics": dict} from that strategy's own BacktestEngine run.
    """
    data = fetch_all()

    bt1_signals, bt1_sizes, bt1_stops = bt1_build_signals(data)
    bt2_signals, bt2_sizes, bt2_stops = bt2_build_signals(data)
    bt3_signals, bt3_sizes, bt3_stops, bt3_regime = bt3_build_signals(data)

    signal_sets = {
        "BT1_Technical": (bt1_signals, bt1_sizes, bt1_stops),
        "BT2_Macro": (bt2_signals, bt2_sizes, bt2_stops),
        "BT3_Regime": (bt3_signals, bt3_sizes, bt3_stops),
    }

    runs = {}
    for label in STRATEGY_LABELS:
        signals, position_sizes, stops = signal_sets[label]
        engine = BacktestEngine(data, initial_capital=INITIAL_CAPITAL)
        results = engine.run(signals, position_sizes, stops)
        metrics = engine.compute_metrics(results)
        runs[label] = {"results": results, "metrics": metrics}

    runs["BT3_Regime"]["results"]["regime"] = bt3_regime.to_numpy()

    return runs


def _determine_winner(values: dict[str, float], higher_wins: bool | None) -> str:
    """Pick the best-scoring strategy label for one metric, or a neutral marker."""
    if higher_wins is None:
        return "—"

    best_label = max(values, key=values.get) if higher_wins else min(values, key=values.get)
    best_value = values[best_label]
    if all(abs(value - best_value) < 1e-9 for value in values.values()):
        return "Tie"
    return best_label


def print_comparison_table(runs: dict[str, dict]) -> None:
    """Print BT1/BT2/BT3 metrics side by side with a per-metric Winner column."""
    label_width = max(len(name) for _, name, _, _ in METRICS) + 2
    col_width = 18
    total_width = label_width + col_width * (len(STRATEGY_LABELS) + 1)

    divider = "=" * total_width
    print(f"\n{divider}")
    print("BT1_Technical vs BT2_Macro vs BT3_Regime — Full Comparison")
    print(divider)
    header = f"{'Metric':<{label_width}}"
    for label in STRATEGY_LABELS:
        header += f"{label:>{col_width}}"
    header += f"{'Winner':>{col_width}}"
    print(header)
    print("-" * total_width)

    for key, name, fmt, higher_wins in METRICS:
        values = {label: runs[label]["metrics"][key] for label in STRATEGY_LABELS}
        row = f"{name:<{label_width}}"
        for label in STRATEGY_LABELS:
            row += f"{fmt.format(values[label]):>{col_width}}"
        row += f"{_determine_winner(values, higher_wins):>{col_width}}"
        print(row)
    print(f"{divider}\n")


def plot_comparison(runs: dict[str, dict]) -> None:
    """Save a 3-panel chart: portfolio value, drawdown, and rolling Sharpe."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, (ax_value, ax_drawdown, ax_sharpe) = plt.subplots(
        3, 1, figsize=(12, 14), facecolor="#fcfcfb", sharex=True
    )

    for ax in (ax_value, ax_drawdown, ax_sharpe):
        ax.set_facecolor("#fcfcfb")
        ax.grid(True, color="#e1e0d9", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors="#898781")

    # Top: portfolio value vs BTC buy & hold.
    for label in STRATEGY_LABELS:
        results = runs[label]["results"]
        ax_value.plot(
            results.index,
            results["portfolio_value"],
            label=label,
            color=STRATEGY_COLORS[label],
            linewidth=2,
        )
    btc_hold = runs[STRATEGY_LABELS[0]]["results"]["btc_hold_value"]
    ax_value.plot(
        btc_hold.index,
        btc_hold,
        label="BTC Buy & Hold",
        color=BTC_HOLD_COLOR,
        linewidth=2,
        linestyle="--",
    )
    ax_value.set_ylabel("Portfolio Value ($)", color="#52514e")
    ax_value.set_title("BT1 vs BT2 vs BT3 vs BTC Buy & Hold", color="#0b0b0b")
    ax_value.legend(frameon=False)

    # Middle: drawdown curves.
    for label in STRATEGY_LABELS:
        results = runs[label]["results"]
        running_max = results["portfolio_value"].cummax()
        drawdown_pct = (results["portfolio_value"] / running_max - 1) * 100
        ax_drawdown.plot(
            results.index, drawdown_pct, label=label, color=STRATEGY_COLORS[label], linewidth=1.5
        )
    ax_drawdown.set_ylabel("Drawdown (%)", color="#52514e")
    ax_drawdown.set_title("Drawdown", color="#0b0b0b")
    ax_drawdown.legend(frameon=False)

    # Bottom: rolling 90-day Sharpe ratio.
    for label in STRATEGY_LABELS:
        results = runs[label]["results"]
        daily_returns = results["returns"]
        rolling_mean = daily_returns.rolling(ROLLING_SHARPE_WINDOW_DAYS).mean()
        rolling_std = daily_returns.rolling(ROLLING_SHARPE_WINDOW_DAYS).std().replace(0, np.nan)
        rolling_sharpe = np.sqrt(TRADING_DAYS_PER_YEAR) * rolling_mean / rolling_std
        ax_sharpe.plot(
            results.index, rolling_sharpe, label=label, color=STRATEGY_COLORS[label], linewidth=1.5
        )
    ax_sharpe.axhline(0, color="#898781", linewidth=0.8)
    ax_sharpe.set_ylabel(f"{ROLLING_SHARPE_WINDOW_DAYS}-Day Rolling Sharpe", color="#52514e")
    ax_sharpe.set_xlabel("Date", color="#52514e")
    ax_sharpe.set_title(f"Rolling {ROLLING_SHARPE_WINDOW_DAYS}-Day Sharpe Ratio", color="#0b0b0b")
    ax_sharpe.legend(frameon=False)

    fig.tight_layout()
    output_path = RESULTS_DIR / "comparison.png"
    fig.savefig(output_path)
    plt.close(fig)

    print(f"Comparison chart saved to {output_path}")


def print_interpretation(runs: dict[str, dict]) -> None:
    """Print a plain-English readout of what each strategy's metrics mean."""
    divider = "=" * 70
    print(f"\n{divider}")
    print("Plain-English Interpretation")
    print(divider)

    for label in STRATEGY_LABELS:
        metrics = runs[label]["metrics"]
        final_value = INITIAL_CAPITAL * (1 + metrics["total_return_pct"] / 100)
        btc_final_value = INITIAL_CAPITAL * (1 + metrics["btc_hold_return_pct"] / 100)
        drawdown_dollars = INITIAL_CAPITAL * abs(metrics["max_drawdown_pct"]) / 100
        beat_hold = metrics["total_return_pct"] > metrics["btc_hold_return_pct"]

        print(f"\n{label}:")
        print(
            f"  - Total return of {metrics['total_return_pct']:.2f}% means a $10,000 starting "
            f"stake would have grown to ${final_value:,.0f}."
        )
        print(
            f"  - Simply holding BTC over the same period would have produced "
            f"${btc_final_value:,.0f} ({metrics['btc_hold_return_pct']:.2f}%), so the strategy "
            f"{'beat' if beat_hold else 'lagged'} buy-and-hold."
        )
        print(
            f"  - A Sharpe ratio of {metrics['sharpe_ratio']:.2f} means returns came with "
            f"{'solid' if metrics['sharpe_ratio'] > 1 else 'weak'} compensation for the volatility "
            f"taken on (above 1.0 is generally considered good, above 2.0 excellent)."
        )
        print(
            f"  - A Sortino ratio of {metrics['sortino_ratio']:.2f} scores returns against only the "
            f"downside volatility, ignoring the upside swings investors don't mind."
        )
        print(
            f"  - Max drawdown of {metrics['max_drawdown_pct']:.2f}% means the strategy lost at most "
            f"${drawdown_dollars:,.0f} on a $10,000 portfolio before recovering."
        )
        print(
            f"  - That worst drawdown took {metrics['max_drawdown_duration_days']} days to recover "
            f"from, i.e. how long the strategy spent underwater at its lowest point."
        )
        print(
            f"  - A Calmar ratio of {metrics['calmar_ratio']:.2f} weighs annualized return against that "
            f"same drawdown — higher means more return earned per unit of worst-case pain."
        )
        print(
            f"  - A win rate of {metrics['win_rate_pct']:.2f}% means roughly "
            f"{round(metrics['win_rate_pct'])} out of every 100 trades closed profitably."
        )
        print(
            f"  - The average trade lasted {metrics['avg_trade_duration_days']:.1f} days across "
            f"{metrics['num_trades']} total trades taken over the backtest period."
        )
    print()


def main() -> None:
    """Run BT1, BT2, and BT3, then print the comparison table, chart, and readout."""
    runs = run_all_backtests()
    print_comparison_table(runs)
    plot_comparison(runs)
    print_interpretation(runs)


if __name__ == "__main__":
    main()
