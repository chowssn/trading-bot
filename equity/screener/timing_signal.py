"""Entry-timing classifier for the equity screener.

Takes the RSI-14D value/direction already computed by `price_filter.py`,
plus (optionally) the RSI-14D series, volume series, and price series for
the same ticker, and turns them into a three-way signal: is a name
showing signs of turning up from oversold (ENTRY), still oversold but not
yet turning (WATCH), or not oversold / falling too hard to call yet
(WAIT).

`_rsi_smoothed_direction()` replaces the old raw 3-day RSI comparison with
a smoothed one: RSI-14D crossed above (or is holding above) its own 5-day
moving average. This is more robust to single-day noise than comparing
three raw RSI points. When no `rsi_14d_series` is supplied (e.g. existing
callers/tests that only pass the scalar value + `rsi_14d_direction`), the
caller-supplied `rsi_14d_direction` is used as-is — this keeps
`classify_timing()` backward compatible with its original 2-argument form.

`_volume_confirms_entry()` is a secondary, non-gating check: buying
pressure (up-day volume) vs. selling pressure (down-day volume) over the
last `market_config.VOLUME_CONFIRMATION_DAYS` days. It never changes the
ENTRY/WATCH/WAIT signal itself, only whether `timing_note` mentions
volume confirmation — see module docstring in `market_config.py`
(VOLUME_CONFIRMATION_DAYS / VOLUME_UP_DOWN_RATIO_MIN).

Evaluation order matters — "deeply oversold and still falling" is checked
before the general oversold WATCH case, so it resolves to WAIT (wait for
stabilization before treating a falling knife as a watch candidate)
rather than WATCH.
"""

import pandas as pd

from equity.config.market_config import VOLUME_CONFIRMATION_DAYS, VOLUME_UP_DOWN_RATIO_MIN

ENTRY = "ENTRY"
WATCH = "WATCH"
WAIT = "WAIT"

DEEP_OVERSOLD_RSI = 25.0
OVERSOLD_RSI = 40.0
NEUTRAL_RSI = 50.0

RSI_5MA_PERIOD = 5


def _rsi_smoothed_direction(rsi_series: pd.Series | None) -> str:
    """RSI 14D vs. its own 5-day moving average — smoothed direction.

    More robust than a raw 3-day point comparison. Returns 'rising',
    'falling', or 'neutral'. 'rising' covers both a fresh crossover above
    the 5MA and an established position above it that's still climbing;
    'falling' is below the 5MA and still dropping. Anything else (fewer
    than 6 data points, or a series that doesn't clearly fit either case)
    is 'neutral'.
    """
    if rsi_series is None or len(rsi_series) < 6:
        return "neutral"

    rsi_5ma = rsi_series.rolling(RSI_5MA_PERIOD).mean()
    current_rsi = rsi_series.iloc[-1]
    current_ma = rsi_5ma.iloc[-1]
    prev_rsi = rsi_series.iloc[-3]
    prev_ma = rsi_5ma.iloc[-3]

    if pd.isna(current_ma) or pd.isna(prev_ma):
        return "neutral"

    # Crossed above the 5MA and holding above it.
    if current_rsi > current_ma and prev_rsi <= prev_ma:
        return "rising"  # fresh crossover
    if current_rsi > current_ma and current_rsi > rsi_series.iloc[-2]:
        return "rising"  # above MA and still rising
    if current_rsi < current_ma and current_rsi < rsi_series.iloc[-2]:
        return "falling"
    return "neutral"


def _volume_confirms_entry(volume_series: pd.Series | None, price_series: pd.Series | None) -> bool | None:
    """Up-day volume / down-day volume over the last VOLUME_CONFIRMATION_DAYS days.

    True if that ratio clears `VOLUME_UP_DOWN_RATIO_MIN` (buying pressure
    outweighs selling pressure), False if it doesn't, None if either
    series is missing or too short to compute — never raises.
    """
    if volume_series is None or price_series is None:
        return None

    n = VOLUME_CONFIRMATION_DAYS
    if len(volume_series) < n + 1 or len(price_series) < n + 1:
        return None

    recent_vol = volume_series.iloc[-n:]
    recent_price = price_series.iloc[-n - 1:]
    returns = recent_price.pct_change().dropna().iloc[-n:]
    if len(returns) < n:
        return None

    up_vol = recent_vol[returns.values > 0].sum()
    down_vol = recent_vol[returns.values <= 0].sum()
    if down_vol == 0:
        return True
    return (up_vol / down_vol) >= VOLUME_UP_DOWN_RATIO_MIN


def classify_timing(
    rsi_14d: float,
    rsi_14d_direction: str,
    rsi_14d_series: pd.Series | None = None,
    volume_series: pd.Series | None = None,
    price_series: pd.Series | None = None,
) -> dict:
    """Classify entry timing from RSI-14D level + direction, optionally refined by series data.

    `rsi_14d_direction` is used as-is when `rsi_14d_series` isn't supplied
    (backward-compatible 2-argument form); otherwise the smoothed
    direction from `_rsi_smoothed_direction()` takes over. Volume
    confirmation, when computable, is noted in `timing_note` on an ENTRY
    signal but never changes ENTRY/WATCH/WAIT itself.

    Returns {'timing_signal': 'ENTRY' | 'WATCH' | 'WAIT', 'timing_note': str}.
    """
    smoothed = _rsi_smoothed_direction(rsi_14d_series) if rsi_14d_series is not None else rsi_14d_direction
    volume_confirms = _volume_confirms_entry(volume_series, price_series)

    if rsi_14d > NEUTRAL_RSI:
        return {
            "timing_signal": WAIT,
            "timing_note": f"RSI14 {rsi_14d:.0f} not yet oversold — no dislocation edge to time an entry off",
        }

    if rsi_14d < DEEP_OVERSOLD_RSI and smoothed == "falling":
        return {
            "timing_signal": WAIT,
            "timing_note": f"RSI14 {rsi_14d:.0f} deeply oversold and still falling — wait for stabilization",
        }

    if rsi_14d <= OVERSOLD_RSI and smoothed == "rising":
        if volume_confirms is True:
            vol_note = " | Volume confirms ✓"
        elif volume_confirms is False:
            vol_note = " | Volume not confirming"
        else:
            vol_note = ""
        return {
            "timing_signal": ENTRY,
            "timing_note": f"RSI14 {rsi_14d:.0f} rising from oversold — momentum turning{vol_note}",
        }

    if rsi_14d <= OVERSOLD_RSI:
        return {
            "timing_signal": WATCH,
            "timing_note": f"RSI14 {rsi_14d:.0f} oversold but not yet turning ({smoothed})",
        }

    if OVERSOLD_RSI < rsi_14d <= NEUTRAL_RSI and smoothed == "rising":
        return {
            "timing_signal": WATCH,
            "timing_note": f"RSI14 {rsi_14d:.0f} approaching oversold zone, rising",
        }

    # Uncovered middle band (e.g. RSI 40-50, neutral/falling) — not spec'd as
    # ENTRY or WAIT, so default to WATCH rather than silently dropping it.
    return {
        "timing_signal": WATCH,
        "timing_note": f"RSI14 {rsi_14d:.0f} in neutral zone ({smoothed})",
    }
