"""Assembles the full daily morning brief from every brief/portfolio/screener module.

`build_morning_brief()` calls each section's fetch + format pair in
sequence (market snapshot, eco calendar, portfolio monitor, performance
tracker, earnings calendar, sector monitor, news triage, screener),
interleaving an AI synthesis (`brief_synthesizer.synthesize_section()`)
after the sections richest in cross-asset signal, and returns a list of
`(text, keyboard)` pairs rather than one big string — each pair is sent as
its own Telegram message by `equity.telegram.bot.send_brief()`/
`scheduled_morning_brief()`, so alert-relevant sections can carry their
own action buttons (discuss a mover, read a thesis-breaking article, page
through a ticker's full headline list).

Each section is wrapped independently in try/except — one broken data
source degrades to an error placeholder for that section, never takes
down the whole brief.

Performance/earnings/sector sit after portfolio monitor and before news
triage: they extend the same holdings-focused picture portfolio monitor
starts (price action -> benchmark-relative performance -> upcoming
earnings -> correlation/concentration risk) before the brief moves on to
qualitative news and then new-idea screening.

`get_regime_adjusted_screener_params()` is a separate utility (not part of
the assembly above): it fetches the current market regime and looks up
`market_config.REGIME_SCREENER_ADJUSTMENTS` so a caller can apply a higher
MIN_QUALITY_SCORE bar on volatile days. `screener.run_screener()` now
accepts `regime_flags` too, but only to adjust the price-filter's RSI
dislocation threshold (see `price_filter._get_rsi_threshold()`) — wiring
this utility's MIN_QUALITY_SCORE override into that call is still future
work.
"""

import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path

from telegram import InlineKeyboardMarkup

from equity.brief import earnings_monitor, eco_calendar, market_snapshot, performance_tracker, sector_monitor
from equity.brief.brief_synthesizer import synthesize_full_brief, synthesize_section
from equity.config import positions, settings
from equity.config.market_config import REGIME_SCREENER_ADJUSTMENTS
from equity.portfolio import monitor, news_triage
from equity.screener import screener
from equity.telegram.formatters import make_article_keyboard, make_main_menu, make_news_actions, make_tickers_keyboard

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


def _run_synthesis(section_name: str, section_text: str, regime_flags: list[str]) -> str:
    """AI synthesis for one section's already-formatted text; never raises."""
    try:
        return synthesize_section(section_name, section_text, regime_flags)
    except Exception as exc:
        logger.exception("Synthesis for section '%s' failed", section_name)
        return f"[Synthesis unavailable: {exc}]"


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


def build_morning_brief() -> list[tuple[str, "InlineKeyboardMarkup | None"]]:
    """Assemble the full morning brief as a list of (text, keyboard) message pairs.

    Header, market snapshot (+ synthesis), eco calendar, portfolio monitor,
    performance tracker (+ synthesis), earnings calendar, sector monitor
    (+ synthesis), news triage (+ headline-pagination nav + synthesis),
    screener, a full-brief synthesis, and a footer — in that order, each
    section independently fault-tolerant (see module docstring).
    """
    sections: list[tuple[str, InlineKeyboardMarkup | None]] = []
    all_text_for_synthesis: list[str] = []

    # Header
    sections.append((format_morning_brief_header(), None))

    # Market snapshot
    try:
        snapshot = market_snapshot.fetch_market_snapshot()
        snap_text = market_snapshot.format_market_snapshot(snapshot)
    except Exception as exc:
        logger.exception("Morning brief section 'Market snapshot' failed")
        snapshot = {}
        snap_text = f"⚠️ Market snapshot unavailable: {exc}"
    regime_flags = snapshot.get("regime_flags", [])
    sections.append((snap_text, None))
    sections.append((f"💡 *Snapshot*\n{_run_synthesis('market_snapshot', snap_text, regime_flags)}", None))
    all_text_for_synthesis.append(snap_text)

    # Eco calendar
    cal_text = _run_section("Eco calendar", lambda: eco_calendar.fetch_eco_calendar(days_ahead=7), eco_calendar.format_eco_calendar)
    sections.append((cal_text, None))
    all_text_for_synthesis.append(cal_text)

    # Portfolio monitor
    try:
        monitor_data = monitor.run_portfolio_monitor()
        mon_text = monitor.format_portfolio_monitor(monitor_data)
    except Exception as exc:
        logger.exception("Morning brief section 'Portfolio monitor' failed")
        monitor_data = {}
        mon_text = f"⚠️ Portfolio monitor unavailable: {exc}"
    alert_tickers = [
        t for t, d in monitor_data.get("positions", {}).items() if d.get("move_flag") in ("LARGE_UP", "LARGE_DOWN")
    ]
    mon_kb = make_tickers_keyboard(alert_tickers) if alert_tickers else None
    sections.append((mon_text, mon_kb))
    all_text_for_synthesis.append(mon_text)

    # Performance
    perf_text = _run_section("Performance tracker", _fetch_performance_data, lambda data: performance_tracker.format_performance_section(*data))
    sections.append((perf_text, None))
    sections.append((f"💡 *Performance*\n{_run_synthesis('performance', perf_text, regime_flags)}", None))
    all_text_for_synthesis.append(perf_text)

    # Earnings
    earn_text = _run_section("Earnings calendar", lambda: earnings_monitor.fetch_earnings_calendar(days_ahead=7), earnings_monitor.format_earnings_section)
    sections.append((earn_text, None))
    all_text_for_synthesis.append(earn_text)

    # Sector monitor
    sector_text = _run_section("Sector monitor", sector_monitor.fetch_sector_data, sector_monitor.format_sector_section)
    sections.append((sector_text, None))
    sections.append((f"💡 *Sectors*\n{_run_synthesis('sector', sector_text, regime_flags)}", None))
    all_text_for_synthesis.append(sector_text)

    # News triage
    try:
        triage = news_triage.run_news_triage(positions.get_all_tickers())
        triage_text = news_triage.format_news_triage(triage)
    except Exception as exc:
        logger.exception("Morning brief section 'News triage' failed")
        triage = {}
        triage_text = f"⚠️ News triage unavailable: {exc}"
    thesis_alert_tickers = [t for t, d in triage.items() if d.get("has_thesis_alert")]
    thesis_articles = [
        {"url": h.get("url", ""), "ticker": ticker}
        for ticker, data in triage.items()
        for h in data.get("headlines", [])
        if h.get("thesis_breaker_match")
    ]
    article_kb = make_article_keyboard(thesis_articles) if thesis_articles else None
    sections.append((triage_text, article_kb))

    all_news_tickers = [t for t, d in triage.items() if d.get("headlines")]
    news_nav_kb = make_news_actions(thesis_alert_tickers, all_news_tickers)
    if news_nav_kb:
        sections.append(("News navigation:", news_nav_kb))
    sections.append((f"💡 *News*\n{_run_synthesis('news', triage_text, regime_flags)}", None))
    all_text_for_synthesis.append(triage_text)

    # Screener
    try:
        df = screener.run_screener()
        screener_text = screener.format_screener_output(df)
        screener_tickers = df["ticker"].tolist() if df is not None and len(df) > 0 else []
    except Exception as exc:
        logger.exception("Morning brief section 'Screener' failed")
        screener_text = f"⚠️ Screener unavailable: {exc}"
        screener_tickers = []
    screen_kb = make_tickers_keyboard(screener_tickers[:6]) if screener_tickers else None
    sections.append((screener_text, screen_kb))
    all_text_for_synthesis.append(screener_text)

    # Full brief synthesis
    try:
        full_synth = synthesize_full_brief("\n\n".join(all_text_for_synthesis), regime_flags)
    except Exception as exc:
        logger.exception("Full-brief synthesis failed")
        full_synth = f"[Full brief synthesis unavailable: {exc}]"
    sections.append((
        f"════════════════════════════════\n"
        f"🎯 MORNING SYNTHESIS\n"
        f"════════════════════════════════\n{full_synth}",
        None,
    ))

    # Footer
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    sections.append((f"Generated {ts}\nReply /discuss TICKER for a deep dive.", make_main_menu()))

    return sections


def save_brief_to_thread(sections: list[tuple[str, "InlineKeyboardMarkup | None"]], thread_manager) -> None:
    """Persist the brief's synthesis sections to a persistent `topic_BRIEF` thread.

    This is what lets `Advisor._get_cross_thread_context()` (see
    equity.telegram.advisor) surface the morning brief in every
    conversation — ticker, macro, portfolio, general — not just a
    `/discuss` about the brief itself. Only the 💡-prefixed per-section
    syntheses and the final 🎯 full-brief synthesis are saved; the raw
    data sections (screener output, eco calendar, etc.) are already
    available live elsewhere and would just bloat this thread's history.

    Content always starts with "Morning Brief — <date>" — the advisor
    relies on that exact prefix to find the actual brief save even if the
    BRIEF thread has since been chatted in directly (it's switchable via
    /switch topic_BRIEF).
    """
    thread_id = "topic_BRIEF"
    thread_manager.get_or_create_thread(thread_id, "topic", "BRIEF")

    synthesis_lines = [f"Morning Brief — {datetime.now().strftime('%Y-%m-%d')}", ""]
    for text, _ in sections:
        if text and ("💡" in text or "🎯" in text):
            synthesis_lines.append(text.strip())
            synthesis_lines.append("")

    consolidated = "\n".join(synthesis_lines).strip()
    thread_manager.add_message(thread_id, "assistant", consolidated)

    # The advisor caches _get_cross_thread_context() for
    # _CROSS_THREAD_CONTEXT_TTL_SECONDS (see equity.telegram.advisor) so it
    # isn't re-querying SQLite on every chat message. That's fine for the
    # scheduled run, but a manually-triggered /brief should be visible to
    # the advisor immediately rather than up to that TTL later — clear the
    # cache here so the next chat message picks up the fresh brief. Local
    # import: advisor.py imports this module (locally, inside a function)
    # to avoid a module-level circular import, so this side stays local too.
    try:
        import equity.telegram.advisor as advisor_module

        advisor_module._cross_thread_context_cache["text"].clear()
        advisor_module._cross_thread_context_cache["timestamp"].clear()
    except Exception:
        logger.debug("save_brief_to_thread: could not invalidate advisor cross-thread cache", exc_info=True)


def _iter_saved_briefs(thread_manager, limit: int = 60):
    """Yield `(date_str, content)` for every brief saved via
    `save_brief_to_thread()`, most recent first.

    Searches backwards through the `topic_BRIEF` thread for messages
    starting with "Morning Brief — " rather than trusting message order/role
    outright — the thread is switchable (`/switch topic_BRIEF`), so if it's
    ever been chatted in directly, some messages are ordinary replies rather
    than brief saves. `limit` bounds how far back to look; a brief runs at
    most a couple of times a day (scheduled + manual /brief reruns), so 60
    messages comfortably covers several weeks of history.

    Shared by `get_last_brief_synthesis()` and `get_recent_briefs()`.
    """
    thread_id = "topic_BRIEF"
    if not thread_manager.get_thread_info(thread_id):
        return

    messages = thread_manager.get_messages_for_api(thread_id, recent_verbatim=limit)
    for m in reversed(messages):
        content = m.get("content", "")
        if m.get("role") == "assistant" and content.startswith("Morning Brief — "):
            date_str = content.splitlines()[0].removeprefix("Morning Brief — ").strip()
            yield date_str, content


def get_last_brief_synthesis(thread_manager) -> tuple[str, str] | None:
    """Return `(date_str, consolidated_text)` for the most recent brief saved
    via `save_brief_to_thread()`, or `None` if the brief has never been run
    (regardless of how long ago — callers that care about staleness compare
    `date_str` to today themselves).

    Shared by `equity.telegram.advisor.Advisor._get_cross_thread_context()`
    and `equity.telegram.bot`'s BRIEF thread view.
    """
    for date_str, content in _iter_saved_briefs(thread_manager, limit=30):
        return date_str, content
    return None


def get_recent_briefs(thread_manager, days_back: int = 7) -> list[tuple[str, str]]:
    """Return `(date_str, content)` for each of the last `days_back` days
    that has a saved brief, most recent first — unlike
    `get_last_brief_synthesis()`, which stops at the single latest one.

    Used by `equity.telegram.advisor.Advisor` to build a recency-weighted
    brief history (full text for today/yesterday, condensed for older days)
    so a conversation still has brief context on a day the brief hasn't run
    yet, without silently treating a week-old brief as current. A day with
    more than one save (e.g. /brief run manually after the scheduled run
    already fired) contributes only its most recent save.
    """
    cutoff = date.today() - timedelta(days=days_back)
    seen_dates: set[str] = set()
    results = []
    for date_str, content in _iter_saved_briefs(thread_manager):
        if date_str in seen_dates:
            continue
        try:
            brief_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            continue
        if brief_date < cutoff:
            break
        seen_dates.add(date_str)
        results.append((date_str, content))
    return results


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
    sections = build_morning_brief()
    elapsed = time.time() - start

    brief_text = "\n\n".join(t for t, _ in sections if t)
    print(brief_text)

    _BRIEFS_DIR.mkdir(parents=True, exist_ok=True)
    brief_path = _BRIEFS_DIR / f"brief_{date.today().isoformat()}.txt"
    try:
        brief_path.write_text(brief_text)
        print(f"\nSaved to {brief_path}")
    except OSError as exc:
        logger.warning("Failed to save brief to %s: %s", brief_path, exc)

    print(f"Total time: {elapsed:.1f}s")
