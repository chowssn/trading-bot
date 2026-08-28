"""Entry-timing classifier for the equity screener.

Takes the RSI-14D value and direction already computed by `price_filter.py`
(no external calls — pure function) and turns them into a simple three-way
signal: is a name showing signs of turning up from oversold (ENTRY), still
oversold but not yet turning (WATCH), or not oversold / falling too hard to
call yet (WAIT).

Evaluation order matters — "deeply oversold and still falling" is checked
before the general oversold/falling WATCH case, so it resolves to WAIT (per
the module's intent: wait for stabilization before treating a falling knife
as a watch candidate) rather than WATCH.
"""

ENTRY = "ENTRY"
WATCH = "WATCH"
WAIT = "WAIT"

DEEP_OVERSOLD_RSI = 25.0
OVERSOLD_RSI = 40.0
NEUTRAL_RSI = 50.0


def classify_timing(rsi_14d: float, rsi_14d_direction: str) -> dict:
    """Classify entry timing from RSI-14D level + direction ('rising', 'falling', 'neutral').

    Returns {'timing_signal': 'ENTRY' | 'WATCH' | 'WAIT', 'timing_note': str}.
    """
    if rsi_14d > NEUTRAL_RSI:
        return {
            "timing_signal": WAIT,
            "timing_note": f"RSI14 {rsi_14d:.0f} not yet oversold — no dislocation edge to time an entry off",
        }

    if rsi_14d < DEEP_OVERSOLD_RSI and rsi_14d_direction == "falling":
        return {
            "timing_signal": WAIT,
            "timing_note": f"RSI14 {rsi_14d:.0f} deeply oversold and still falling — wait for stabilization",
        }

    if rsi_14d <= OVERSOLD_RSI and rsi_14d_direction == "rising":
        return {
            "timing_signal": ENTRY,
            "timing_note": f"RSI14 {rsi_14d:.0f} rising from oversold — momentum turning",
        }

    if rsi_14d <= OVERSOLD_RSI and rsi_14d_direction in ("neutral", "falling"):
        return {
            "timing_signal": WATCH,
            "timing_note": f"RSI14 {rsi_14d:.0f} oversold but not yet turning ({rsi_14d_direction})",
        }

    if OVERSOLD_RSI < rsi_14d <= NEUTRAL_RSI and rsi_14d_direction == "rising":
        return {
            "timing_signal": WATCH,
            "timing_note": f"RSI14 {rsi_14d:.0f} approaching oversold zone, rising",
        }

    # Uncovered middle band (e.g. RSI 40-50, neutral/falling) — not spec'd as
    # ENTRY or WAIT, so default to WATCH rather than silently dropping it.
    return {
        "timing_signal": WATCH,
        "timing_note": f"RSI14 {rsi_14d:.0f} in neutral zone ({rsi_14d_direction})",
    }
