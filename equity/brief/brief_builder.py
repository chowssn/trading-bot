"""Assembles the full daily morning brief from every brief/portfolio/screener module.

`build_morning_brief()` calls each section's fetch + format pair in
sequence (market snapshot, eco calendar, portfolio monitor, performance
tracker, earnings calendar, sector monitor, news triage, screener) and
stitches the results together. Each section is wrapped independently in
try/except — one broken data source degrades to an error placeholder for
that section, never takes down the whole brief.

Performance/earnings/sector sit after portfolio monitor and before news
triage: they extend the same holdings-focused picture portfolio monitor
starts (price action -> benchmark-relative performance -> upcoming
earnings -> correlation/concentration risk) before the brief moves on to
qualitative news and then new-idea screening.

`get_regime_adjusted_screener_params()` is a separate utility (not part of
the assembly above): it fetches the current market regime and looks up
`market_config.REGIME_SCREENER_ADJUSTMENTS` so the screener can apply a
higher quality bar on volatile days. Wiring it into `screener.run_screener()`
is future work — that function doesn't yet take screener-param overrides.
"""

import logging
import time
from datetime import date, datetime
from pathlib import Path

from equity.brief import earnings_monitor, eco_calendar, market_snapshot, performance_tracker, sector_monitor
from equity.config import positions, settings
from equity.config.market_config import REGIME_SCREENER_ADJUSTMENTS
from equity.portfolio import monitor, news_triage
from equity.screener import screener

logger = logging.getLogger(__name__)

_DIVIDER_HEAVY = "════════════════════════════════"

_BRIEFS_DIR = Path(__file__).resolve().parents[1] / "data" / "briefs"


def format_morning_brief_header() -> str:
    today = date.today().strftime("%a %b %d")
    return f"{_DIVIDER_HEAVY}\n🌅 MORNING BRIEF — {today}\n{_DIVIDER_HEAVY}"


def _run_section(name: str, fetch_fn, format_fn) -> str:
    """Fetch + format one brief section; never raises — returns an error placeholder instead."""
    try:
        return format_fn(fetch_fn())
    except Exception as exc:
        logger.exception("Morning brief section '%s' failed", name)
        return f"⚠️ {name} unavailable: {exc}"


def _fetch_performance_data() -> tuple[dict, dict, dict, dict]:
    """Bundle performance_tracker's four fetches into one tuple for `_run_section()`.

    `format_performance_section()` takes four separate dicts rather than
    one — this just gives `_run_section()` a single fetch_fn to call, in
    keeping with its (name, fetch_fn, format_fn) contract.
    """
    benchmark_data = performance_tracker.fetch_benchmark_performance()
    portfolio_data = performance_tracker.fetch_portfolio_performance()
    relative_data = performance_tracker.fetch_position_relative_performance(benchmark_data)
    spot_data = performance_tracker.fetch_spot_prices(benchmark_data)
    return benchmark_data, portfolio_data, relative_data, spot_data


def build_morning_brief() -> str:
    """Assemble the full morning brief: header, market snapshot, eco calendar, portfolio,
    performance tracker, earnings calendar, sector monitor, news triage, screener, and a
    footer — in that order, each section independently fault-tolerant (see module docstring).
    """
    sections = [format_morning_brief_header()]

    sections.append(_run_section(
        "Market snapshot", market_snapshot.fetch_market_snapshot, market_snapshot.format_market_snapshot,
    ))
    sections.append(_run_section(
        "Eco calendar", eco_calendar.fetch_eco_calendar, eco_calendar.format_eco_calendar,
    ))
    sections.append(_run_section(
        "Portfolio monitor", monitor.run_portfolio_monitor, monitor.format_portfolio_monitor,
    ))
    sections.append(_run_section(
        "Performance tracker",
        _fetch_performance_data,
        lambda data: performance_tracker.format_performance_section(*data),
    ))
    sections.append(_run_section(
        "Earnings calendar", earnings_monitor.fetch_earnings_calendar, earnings_monitor.format_earnings_section,
    ))
    sections.append(_run_section(
        "Sector monitor", sector_monitor.fetch_sector_data, sector_monitor.format_sector_section,
    ))
    sections.append(_run_section(
        "News triage",
        lambda: news_triage.run_news_triage(positions.get_all_tickers()),
        news_triage.format_news_triage,
    ))
    sections.append(_run_section(
        "Screener", screener.run_screener, screener.format_screener_output,
    ))

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sections.append(f"Generated {timestamp}\nReply /discuss TICKER for a deep dive on any name above.")

    return "\n\n".join(sections)


def get_regime_adjusted_screener_params() -> dict:
    """Regime-aware screener parameter overrides, keyed off today's active regime flags.

    Fetches the current market snapshot's `regime_flags` and looks each one
    up in `market_config.REGIME_SCREENER_ADJUSTMENTS`. When multiple active
    flags carry a `min_quality_score_override`, the highest (most
    conservative) one wins; every matching flag's `note` is collected.
    Falls back to `settings.MIN_QUALITY_SCORE` with no notes if the market
    snapshot fetch fails or no active flag has an adjustment defined —
    never raises.
    """
    params = {
        "min_quality_score": settings.MIN_QUALITY_SCORE,
        "active_regime_flags": [],
        "notes": [],
    }

    try:
        snapshot = market_snapshot.fetch_market_snapshot()
    except Exception as exc:
        logger.warning("Could not fetch market snapshot for regime-adjusted screener params: %s", exc)
        return params

    regime_flags = snapshot.get("regime_flags", [])
    params["active_regime_flags"] = regime_flags

    overrides = []
    for flag in regime_flags:
        adjustment = REGIME_SCREENER_ADJUSTMENTS.get(flag)
        if not adjustment:
            continue
        if "min_quality_score_override" in adjustment:
            overrides.append(adjustment["min_quality_score_override"])
        if adjustment.get("note"):
            params["notes"].append(adjustment["note"])

    if overrides:
        params["min_quality_score"] = max(overrides)

    return params


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    start = time.time()
    brief = build_morning_brief()
    elapsed = time.time() - start

    print(brief)

    _BRIEFS_DIR.mkdir(parents=True, exist_ok=True)
    brief_path = _BRIEFS_DIR / f"brief_{date.today().isoformat()}.txt"
    try:
        brief_path.write_text(brief)
        print(f"\nSaved to {brief_path}")
    except OSError as exc:
        logger.warning("Failed to save brief to %s: %s", brief_path, exc)

    print(f"Total time: {elapsed:.1f}s")
