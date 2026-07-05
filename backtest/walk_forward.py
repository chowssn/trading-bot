"""walk_forward.py — walk-forward validation of BT5's signal.

Splits the full BTC-USDC history into two disjoint windows:

- Training window: the first 24 months. Used only to sanity-check that
  BT5's signal parameters produce reasonable behavior in-sample.
- Test window: the final 18 months, held out completely from the
  training window. BT5's exact signal logic (`bt5_filtered.build_signals`)
  is re-run from scratch on this slice alone — no indicator state or
  portfolio state carries over from the training period — so this is
  an honest out-of-sample test.

Metrics are reported separately for the training period, the test
period, and the full period (run once, end to end, for reference).
Finally, the test-period Sharpe is compared to the training-period
Sharpe: a drop of more than 50% is flagged as a potential overfitting
warning.
"""

import pandas as pd
from dateutil.relativedelta import relativedelta

from backtest.bt5_filtered import build_signals
from backtest.data_fetcher import fetch_all
from backtest.engine import BacktestEngine

INITIAL_CAPITAL = 10000.0
COMMISSION_RATE = 0.001

TRAIN_MONTHS = 24
TEST_MONTHS = 18

OVERFIT_SHARPE_RETENTION_THRESHOLD = 0.50


def split_windows(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Slice `data` into a first-N-month training window and a held-out,
    final-N-month test window.

    The two windows are computed independently from the start and end of
    the full history, so the test window shares no rows with training
    and never influences it.
    """
    start = data.index[0]
    end = data.index[-1]

    train_end = start + relativedelta(months=TRAIN_MONTHS)
    test_start = end - relativedelta(months=TEST_MONTHS)

    train = data.loc[start:train_end]
    test = data.loc[test_start:end]
    return train, test


def run_window(data: pd.DataFrame) -> tuple[dict, BacktestEngine, pd.DataFrame]:
    """Run BT5's exact signal logic on `data`, starting from a fresh portfolio.

    Signals are computed from `data` alone, so a window that starts after
    the beginning of history has no lookback into rows outside itself.
    """
    signals, position_sizes, stops, *_ = build_signals(data)
    engine = BacktestEngine(
        data, initial_capital=INITIAL_CAPITAL, commission_rate=COMMISSION_RATE
    )
    results = engine.run(signals, position_sizes, stops)
    metrics = engine.compute_metrics(results)
    return metrics, engine, results


def print_window_summary(label: str, data: pd.DataFrame) -> None:
    """Print the date range and row count of a window before its results."""
    print(
        f"\n{label} window: {data.index[0].date()} to {data.index[-1].date()} "
        f"({len(data)} days)"
    )


def print_overfitting_check(train_metrics: dict, test_metrics: dict) -> None:
    """Compare test-period Sharpe against training-period Sharpe and flag
    a drop of more than 50% as a potential overfitting warning.
    """
    train_sharpe = train_metrics["sharpe_ratio"]
    test_sharpe = test_metrics["sharpe_ratio"]

    divider = "=" * 70
    print(f"\n{divider}")
    print("Walk-Forward Overfitting Check")
    print(divider)
    print(f"Training Sharpe Ratio: {train_sharpe:.2f}")
    print(f"Test Sharpe Ratio:     {test_sharpe:.2f}")

    if train_sharpe <= 0:
        print(
            "Training Sharpe was <= 0, so 'retained % of training Sharpe' isn't a "
            "meaningful ratio."
        )
        if test_sharpe < train_sharpe:
            print("Test Sharpe is even lower than training Sharpe — treat results with caution.")
        else:
            print("Test Sharpe did not fall further than training Sharpe.")
    else:
        retained_fraction = test_sharpe / train_sharpe
        print(f"Test Sharpe retained {retained_fraction * 100:.1f}% of training Sharpe.")
        if retained_fraction < OVERFIT_SHARPE_RETENTION_THRESHOLD:
            print(
                "WARNING: Test Sharpe dropped more than 50% vs. training Sharpe — "
                "potential overfitting."
            )
        else:
            print("OK: Test Sharpe stayed within 50% of training Sharpe.")
    print(f"{divider}\n")


def main() -> None:
    """Fetch BTC+macro data, run BT5 walk-forward, and print metrics + overfitting check."""
    data = fetch_all()

    train_data, test_data = split_windows(data)

    print_window_summary("Training", train_data)
    train_metrics, train_engine, _train_results = run_window(train_data)
    train_engine.print_metrics(train_metrics, label="BT5_Filtered — Training")

    print_window_summary("Test (out-of-sample)", test_data)
    test_metrics, test_engine, _test_results = run_window(test_data)
    test_engine.print_metrics(test_metrics, label="BT5_Filtered — Test")

    print_window_summary("Full", data)
    full_metrics, full_engine, _full_results = run_window(data)
    full_engine.print_metrics(full_metrics, label="BT5_Filtered — Full Period")

    print_overfitting_check(train_metrics, test_metrics)


if __name__ == "__main__":
    main()
