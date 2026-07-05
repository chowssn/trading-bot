"""Core backtest simulation engine shared by all backtest/ strategies.

Every entry and exit executes at the *next* day's open — never at the
signal bar's own close or low — so results are free of lookahead bias.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252
CALENDAR_DAYS_PER_YEAR = 365

RESULTS_DIR = Path(__file__).resolve().parent / "results"

STRATEGY_COLOR = "#2a78d6"
BENCHMARK_COLOR = "#eb6834"


class BacktestEngine:
    """Simulates daily execution of a long/flat strategy against BTC data.

    Holds no strategy logic itself — callers supply precomputed signal,
    position-size, and stop series, and the engine handles order timing,
    portfolio accounting, and performance metrics.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        initial_capital: float = 10000.0,
        commission_rate: float = 0.001,
    ) -> None:
        """Store the market data and starting capital for the simulation.

        Args:
            data: Combined OHLCV + macro DataFrame from data_fetcher.fetch_all(),
                indexed by date, with at least open/high/low/close columns.
            initial_capital: Starting cash balance in quote currency.
            commission_rate: Fraction of trade value charged as commission on
                every entry and exit (default 0.001 = 0.1%, Coinbase taker fee).
        """
        self.data = data
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.trades: list[dict] = []

    def run(
        self,
        signals: pd.Series,
        position_sizes: pd.Series,
        stops: pd.Series,
    ) -> pd.DataFrame:
        """Simulate daily trading and return a portfolio history DataFrame.

        A long position is opened at the next day's open once `signals`
        turns 1, and closed at the next day's open once `signals` drops
        back to 0 or the current day's low breaches the active stop.

        Args:
            signals: 1 (long) or 0 (flat) per day, indexed by date.
            position_sizes: Fraction of available cash (0.0-1.0) to commit
                on entry, indexed by date.
            stops: Stop price active for a held position, indexed by date;
                NaN means no stop is active.

        Returns:
            DataFrame indexed like `data` with columns: portfolio_value,
            returns, btc_hold_value, signal, position_size, in_trade.
        """
        dates = self.data.index
        signals = signals.reindex(dates).fillna(0)
        position_sizes = position_sizes.reindex(dates).fillna(0.0)
        stops = stops.reindex(dates)

        opens = self.data["open"]
        lows = self.data["low"]
        closes = self.data["close"]

        cash = self.initial_capital
        btc_position = 0.0
        trade_active = False
        entry_price = None
        entry_date = None
        entry_trade_value = 0.0
        pending_action = None
        pending_size = 0.0
        pending_exit_reason = None

        self.trades = []
        portfolio_values = []
        in_trade_flags = []

        for i, date in enumerate(dates):
            open_price = opens.iloc[i]
            low_price = lows.iloc[i]
            close_price = closes.iloc[i]

            if pending_action == "enter" and not trade_active:
                trade_value = cash * pending_size
                btc_position = trade_value / open_price
                cash -= trade_value
                cash -= self.commission_rate * trade_value
                trade_active = True
                entry_price = open_price
                entry_date = date
                entry_trade_value = trade_value
            elif pending_action == "exit" and trade_active:
                trade_value = btc_position * open_price
                cash += trade_value
                cash -= self.commission_rate * trade_value
                self.trades.append(
                    {
                        "entry_date": entry_date,
                        "exit_date": date,
                        "entry_price": entry_price,
                        "exit_price": open_price,
                        "return_pct": (open_price / entry_price - 1) * 100,
                        "duration_days": (date - entry_date).days,
                        "exit_reason": pending_exit_reason,
                        "pnl_dollars": trade_value * (1 - self.commission_rate)
                        - entry_trade_value * (1 + self.commission_rate),
                    }
                )
                btc_position = 0.0
                trade_active = False
                entry_price = None
                entry_date = None
            pending_action = None
            pending_exit_reason = None

            portfolio_values.append(cash + btc_position * close_price)
            in_trade_flags.append(trade_active)

            todays_signal = signals.iloc[i]
            todays_stop = stops.iloc[i]
            stop_hit = trade_active and pd.notna(todays_stop) and low_price <= todays_stop

            if trade_active and (todays_signal == 0 or stop_hit):
                pending_action = "exit"
                pending_exit_reason = "stop_hit" if stop_hit else "technical_exit"
            elif not trade_active and todays_signal == 1:
                pending_action = "enter"
                pending_size = position_sizes.iloc[i]

        if trade_active:
            last_date = dates[-1]
            last_close = closes.iloc[-1]
            final_trade_value = btc_position * last_close
            self.trades.append(
                {
                    "entry_date": entry_date,
                    "exit_date": last_date,
                    "entry_price": entry_price,
                    "exit_price": last_close,
                    "return_pct": (last_close / entry_price - 1) * 100,
                    "duration_days": (last_date - entry_date).days,
                    "exit_reason": "end_of_data",
                    "pnl_dollars": final_trade_value - entry_trade_value * (1 + self.commission_rate),
                }
            )

        btc_hold_units = self.initial_capital / opens.iloc[0]
        btc_hold_value = btc_hold_units * closes

        results = pd.DataFrame(index=dates)
        results["portfolio_value"] = portfolio_values
        results["returns"] = results["portfolio_value"].pct_change()
        results["btc_hold_value"] = btc_hold_value.to_numpy()
        results["signal"] = signals.to_numpy()
        results["position_size"] = position_sizes.to_numpy()
        results["in_trade"] = in_trade_flags

        return results

    def compute_metrics(self, results: pd.DataFrame) -> dict:
        """Compute performance metrics from a `run()` results DataFrame.

        Args:
            results: Output of `run()` for the same engine instance (so
                that `self.trades` matches the run being scored).

        Returns:
            Dict of performance metrics; see method docstrings on the
            engine for the full list of keys.
        """
        daily_returns = results["returns"].dropna()

        total_return_pct = (
            results["portfolio_value"].iloc[-1] / self.initial_capital - 1
        ) * 100
        btc_hold_return_pct = (
            results["btc_hold_value"].iloc[-1] / self.initial_capital - 1
        ) * 100

        return_std = daily_returns.std()
        sharpe_ratio = (
            np.sqrt(TRADING_DAYS_PER_YEAR) * daily_returns.mean() / return_std
            if return_std
            else 0.0
        )

        downside_std = daily_returns[daily_returns < 0].std()
        sortino_ratio = (
            np.sqrt(TRADING_DAYS_PER_YEAR) * daily_returns.mean() / downside_std
            if downside_std
            else 0.0
        )

        running_max = results["portfolio_value"].cummax()
        drawdown = results["portfolio_value"] / running_max - 1
        max_drawdown_pct = drawdown.min() * 100

        underwater = results["portfolio_value"] < running_max
        new_peak_groups = (~underwater).cumsum()
        streak_lengths = underwater.groupby(new_peak_groups).sum()
        max_drawdown_duration_days = int(streak_lengths.max()) if len(streak_lengths) else 0

        num_days = (results.index[-1] - results.index[0]).days
        annualized_return = (
            (results["portfolio_value"].iloc[-1] / self.initial_capital)
            ** (CALENDAR_DAYS_PER_YEAR / num_days)
            - 1
            if num_days
            else 0.0
        )
        calmar_ratio = (
            annualized_return / abs(max_drawdown_pct / 100) if max_drawdown_pct else 0.0
        )

        num_trades = len(self.trades)
        if num_trades:
            winning_trades = sum(1 for t in self.trades if t["return_pct"] > 0)
            win_rate_pct = winning_trades / num_trades * 100
            avg_trade_duration_days = sum(t["duration_days"] for t in self.trades) / num_trades
        else:
            win_rate_pct = 0.0
            avg_trade_duration_days = 0.0

        total_profit_dollars = self.initial_capital * (total_return_pct / 100)
        if num_trades and total_profit_dollars:
            top_3_pnl_dollars = sum(
                sorted((t["pnl_dollars"] for t in self.trades), reverse=True)[:3]
            )
            pct_return_from_top_3_trades = top_3_pnl_dollars / total_profit_dollars * 100
        else:
            pct_return_from_top_3_trades = 0.0

        return {
            "total_return_pct": total_return_pct,
            "btc_hold_return_pct": btc_hold_return_pct,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown_pct": max_drawdown_pct,
            "max_drawdown_duration_days": max_drawdown_duration_days,
            "calmar_ratio": calmar_ratio,
            "win_rate_pct": win_rate_pct,
            "avg_trade_duration_days": avg_trade_duration_days,
            "num_trades": num_trades,
            "pct_return_from_top_3_trades": pct_return_from_top_3_trades,
        }

    def print_metrics(self, metrics: dict, label: str = "Strategy") -> None:
        """Print a formatted summary table for a metrics dict.

        Args:
            metrics: Output of `compute_metrics()`.
            label: Name shown in the table header.
        """
        rows = [
            ("Total Return", f"{metrics['total_return_pct']:.2f}%"),
            ("BTC Buy & Hold Return", f"{metrics['btc_hold_return_pct']:.2f}%"),
            ("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}"),
            ("Sortino Ratio", f"{metrics['sortino_ratio']:.2f}"),
            ("Max Drawdown", f"{metrics['max_drawdown_pct']:.2f}%"),
            ("Max Drawdown Duration", f"{metrics['max_drawdown_duration_days']} days"),
            ("Calmar Ratio", f"{metrics['calmar_ratio']:.2f}"),
            ("Win Rate", f"{metrics['win_rate_pct']:.2f}%"),
            ("Avg Trade Duration", f"{metrics['avg_trade_duration_days']:.1f} days"),
            ("Num Trades", f"{metrics['num_trades']}"),
            (
                "Pct Return From Top 3 Trades",
                f"{metrics['pct_return_from_top_3_trades']:.2f}%",
            ),
        ]
        label_width = max(len(name) for name, _ in rows) + 2

        divider = "=" * 50
        print(f"\n{divider}")
        print(f"{label} — Backtest Results")
        print(divider)
        for name, value in rows:
            print(f"{name:<{label_width}}{value}")
        print(f"{divider}\n")

    def plot_results(self, results: pd.DataFrame, label: str = "Strategy") -> None:
        """Plot strategy portfolio value against BTC buy-and-hold and save it.

        Args:
            results: Output of `run()`.
            label: Series name and output filename stem
                (backtest/results/{label}.png).
        """
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(12, 6), facecolor="#fcfcfb")
        ax.set_facecolor("#fcfcfb")

        ax.plot(
            results.index,
            results["portfolio_value"],
            label=label,
            color=STRATEGY_COLOR,
            linewidth=2,
        )
        ax.plot(
            results.index,
            results["btc_hold_value"],
            label="BTC Buy & Hold",
            color=BENCHMARK_COLOR,
            linewidth=2,
        )

        ax.set_xlabel("Date", color="#52514e")
        ax.set_ylabel("Portfolio Value ($)", color="#52514e")
        ax.set_title(f"{label} vs BTC Buy & Hold", color="#0b0b0b")
        ax.grid(True, color="#e1e0d9", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors="#898781")
        ax.legend(frameon=False)

        fig.tight_layout()
        fig.savefig(RESULTS_DIR / f"{label}.png")
        plt.close(fig)
