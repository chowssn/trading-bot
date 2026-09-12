"""Assembles the daily morning brief from every brief/portfolio/screener module.

`build_morning_brief()` assembles 4 self-contained data sections — Global
Markets, Economic Calendar, Portfolio Status, News & Signals (see each
`_format_sectionN_*()` below) — plus a single closing Morning Synthesis
that covers all four (`brief_synthesizer.synthesize_full_brief()`). This
replaces the earlier per-section synthesis calls (one
`synthesize_section()` call after sections 1, 3, and 4) with one AI call
at the end of the brief — cheaper, and the only call with visibility
across every section, so it's the one place cross-section connections
(a geopolitical event in Section 1 moving a Section 3 position) can
actually get made. The economic calendar is data-only either way — no
synthesis, it's self-explanatory. A table of contents with per-section
re-send buttons (`make_brief_toc_keyboard()`) follows the header, since
this section list is now longer to scroll back through than the sections
themselves.

Returns a list of `(text, keyboard)` pairs rather than one big string —
each pair is sent as its own Telegram message by
`equity.telegram.bot.send_brief()`/`scheduled_morning_brief()`, so
alert-relevant sections can carry their own action buttons (discuss a
mover, read a thesis-breaking article, page through a ticker's full
headline list, or re-open a section from the table of contents).

Each section is wrapped independently in try/except — one broken data
source degrades to an error placeholder for that section, never takes
down the whole brief.

`_run_section()`/`_run_synthesis()`/`_fetch_performance_data()` are the
older per-sub-section helpers `build_morning_brief()` used before the
5-section reorg. Kept (unused by the current assembly) rather than
deleted — nothing else in the codebase calls them, but they're small and
this module's job is exactly this kind of glue.

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
from equity.brief.brief_synthesizer import (
    _parse_and_persist_monitoring,
    synthesize_full_brief,
    synthesize_section,
)
from equity.config import positions, settings
from equity.config.market_config import REGIME_SCREENER_ADJUSTMENTS
from equity.data.monitoring import deduplicate_monitoring, load_monitoring
from equity.portfolio import monitor, news_triage
from equity.screener import screener
from equity.telegram.formatters import (
    make_article_keyboard,
    make_brief_toc_keyboard,
    make_main_menu,
    make_news_actions,
    make_tickers_keyboard,
)

logger = logging.getLogger(__name__)

_DIVIDER_HEAVY = "════════════════════════════════"
_DIVIDER_LIGHT = "━━━━━━━━━━━━━━━━━━━━━━━━"

_BRIEFS_DIR = Path(__file__).resolve().parents[1] / "data" / "briefs"


def format_morning_brief_header() -> str:
    today = date.today().strftime("%a %b %d")
    return f"{_DIVIDER_HEAVY}\n🌅 MORNING BRIEF — {today}\n{_DIVIDER_HEAVY}"


def _run_section(name: str, fetch_fn, format_fn) -> str:
    """Fetch + format one brief section; never raises — returns an error placeholder instead.

    Logs how long the section took (or how long it ran before failing) to
    brief.log, so slow or broken sections are visible without re-running
    the whole brief — see equity/config/logging_config.py.
    """
    start = time.time()
    try:
        result = format_fn(fetch_fn())
        logger.info("Brief section [%s] completed in %.1fs", name, time.time() - start)
        return result
    except Exception as exc:
        logger.error("Brief section [%s] FAILED after %.1fs: %s", name, time.time() - start, exc, exc_info=True)
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


def _format_section1_global_markets(snapshot: dict, global_signals: dict) -> str:
    """Combines market_snapshot + global_signals into Section 1, plus the
    web-searched macro intelligence narrative layer (geopolitical/central
    bank/economic release/overnight session/analyst action context).

    `fetch_macro_intelligence()` is isolated in its own try/except here,
    separate from build_morning_brief()'s outer per-section try/except —
    a web-search or Claude outage should cost the brief its narrative
    overlay, not the quantitative snapshot underneath it. The macro intel
    block is appended *after* the quantitative data (not before it) so
    `synthesize_section()`'s `section_data[:...]` truncation, if it has to
    cut anything, cuts the newer narrative addition rather than the
    numbers the SIGNAL/POSITIONS AFFECTED synthesis actually depends on.
    """
    intel_text = ""
    try:
        intel = market_snapshot.fetch_macro_intelligence(regime_flags=snapshot.get("regime_flags", []))
        intel_text = market_snapshot.format_macro_intelligence(intel)
    except Exception:
        logger.exception("Section 1: fetch_macro_intelligence failed — continuing without it")

    parts = [
        "🌍 GLOBAL MARKETS",
        _DIVIDER_LIGHT,
        market_snapshot.format_market_snapshot(snapshot),
        "",
        market_snapshot.format_global_signals(global_signals),
    ]
    if intel_text:
        parts.extend(["", intel_text])
    return "\n".join(parts)


def _format_section2_eco_calendar(cal: dict, earnings: dict) -> str:
    """Eco calendar + earnings for held positions only. No synthesis — data only."""
    parts = [
        "📅 ECONOMIC CALENDAR",
        _DIVIDER_LIGHT,
        eco_calendar.format_eco_calendar(cal),
    ]
    earnings_text = earnings_monitor.format_earnings_section(earnings)
    if earnings_text and "no upcoming" not in earnings_text.lower():
        parts.extend(["", earnings_text])
    return "\n".join(parts)


def _format_section3_portfolio_status(monitor_data, perf_data, rel_data, spot_data, port_data, sector_data) -> str:
    """Portfolio monitor + benchmarks + sectors + concentration. Calls existing format functions."""
    parts = [
        "💼 PORTFOLIO STATUS",
        _DIVIDER_LIGHT,
        monitor.format_portfolio_monitor(monitor_data),
        "",
        performance_tracker.format_performance_section(perf_data, port_data, rel_data, spot_data),
        "",
        sector_monitor.format_sector_section(sector_data),
    ]
    return "\n".join(parts)


def _format_section4_news_signals(triage: dict, df_screener) -> str:
    """News triage + screener results."""
    parts = [
        "📰 NEWS & SIGNALS",
        _DIVIDER_LIGHT,
        news_triage.format_news_triage(triage),
    ]
    if df_screener is not None and len(df_screener) > 0:
        parts.extend(["", "📊 SCREENER", screener.format_screener_output(df_screener)])
    else:
        parts.append("\n📊 SCREENER: No names passed all filters today.")
    return "\n".join(parts)


def _auto_cleanup_monitoring() -> int:
    """Resolves stale active monitoring items at the start of each brief run:
    low priority items older than 7 days, medium priority older than 21
    days. High priority items never auto-expire — they wait for an explicit
    `/dismiss` or a synthesis-driven supersession (see
    `equity.data.monitoring._supersedes()`).

    Returns count of items resolved.
    """
    from equity.data.monitoring import _load_all, _save_all

    data = _load_all()
    today = date.today()
    resolved = 0

    stale_thresholds = {"low": 7, "medium": 21, "high": None}

    for item in data.get("items", []):
        if item.get("status") != "active":
            continue

        threshold_days = stale_thresholds.get(item.get("priority", "medium"))
        if threshold_days is None:
            continue

        try:
            added = date.fromisoformat(item.get("added_date", str(today)))
        except ValueError:
            continue

        if (today - added).days >= threshold_days:
            item["status"] = "resolved"
            item.setdefault("notes", []).append(
                f"Auto-resolved {today}: exceeded {threshold_days}d auto-expiry "
                f'for {item.get("priority", "medium")} priority'
            )
            resolved += 1

    if resolved > 0:
        _save_all(data)
        logger.info("_auto_cleanup_monitoring: auto-resolved %d stale items", resolved)

    return resolved


def build_morning_brief() -> list[tuple[str, "InlineKeyboardMarkup | None"]]:
    """Assemble the morning brief as a list of (text, keyboard) message pairs.

    Header, table of contents, then:
      1. Global Markets    — market snapshot + global signals, no per-section synthesis
      2. Economic Calendar — week view + FOMC proximity + held-position earnings, no synthesis
      3. Portfolio Status  — monitor + benchmarks + sectors, no per-section synthesis
      4. News & Signals    — news triage + screener, no per-section synthesis
      5. Morning Synthesis — single end-of-brief synthesis covering all four sections above
    and a footer. Each section is independently fault-tolerant: an
    exception degrades to a warning placeholder for that section only and
    never takes down the rest of the brief. Synthesis happens exactly once,
    at the end (see `brief_synthesizer.SYNTHESIS_MAX_TOKENS["full_brief"]`
    for its token budget) — see module docstring for why the per-section
    calls were removed.
    """
    sections: list[tuple[str, InlineKeyboardMarkup | None]] = []
    all_text: list[str] = []
    regime_flags: list[str] = []

    # ── MONITORING HOUSEKEEPING ──────────────────────────────────
    # Resolve stale items and merge duplicates before anything below reads
    # `load_monitoring()`, so stale/duplicate items never leak into a
    # section's ACTIVE MONITORING context or the monitoring keyboard.
    try:
        cleaned = _auto_cleanup_monitoring()
        if cleaned > 0:
            logger.info("Morning brief: auto-cleaned %d stale monitoring items", cleaned)
        deduped = deduplicate_monitoring()
        if deduped > 0:
            logger.info("Morning brief: deduplicated %d monitoring items", deduped)
    except Exception:
        logger.exception("Morning brief: monitoring housekeeping failed — continuing with brief")

    # ── HEADER ──────────────────────────────────────────────────
    sections.append((format_morning_brief_header(), None))

    # ── TABLE OF CONTENTS ────────────────────────────────────────
    # Private chats can't use message links, so this is a callback-driven
    # re-send (see equity.telegram.bot._send_brief_section()) rather than a
    # jump-to-message.
    toc_text = (
        f"📋 *Morning Brief — {datetime.now().strftime('%a %b %d, %Y')}*\n"
        f"Tap any section to view it again:"
    )
    sections.append((toc_text, make_brief_toc_keyboard()))

    # ── SECTION 1: GLOBAL MARKETS ───────────────────────────────
    # Consolidates: market snapshot (rates/FX/commodities/equity futures)
    # + global signals (volatility/crypto/international/credit/cross-asset).
    try:
        snapshot = market_snapshot.fetch_market_snapshot()
        regime_flags = snapshot.get("regime_flags", [])
        global_signals = market_snapshot.fetch_global_signals()

        s1_text = _format_section1_global_markets(snapshot, global_signals)
        sections.append((s1_text, None))
        all_text.append(s1_text)
    except Exception as exc:
        logger.error("Section 1 (Global Markets) FAILED: %s", exc, exc_info=True)
        sections.append((f"⚠️ Global markets data unavailable: {exc}", None))

    # ── SECTION 2: ECONOMIC CALENDAR ────────────────────────────
    # No synthesis — data only, self-explanatory.
    try:
        cal = eco_calendar.fetch_eco_calendar(days_ahead=7)
        earnings = earnings_monitor.fetch_earnings_calendar(days_ahead=14)
        s2_text = _format_section2_eco_calendar(cal, earnings)
        sections.append((s2_text, None))
        all_text.append(s2_text)
    except Exception as exc:
        logger.error("Section 2 (Economic Calendar) FAILED: %s", exc, exc_info=True)
        sections.append((f"⚠️ Economic calendar unavailable: {exc}", None))

    # ── SECTION 3: PORTFOLIO STATUS ──────────────────────────────
    # Consolidates: portfolio monitor + benchmark/relative performance +
    # sector exposure/concentration.
    try:
        monitor_data = monitor.run_portfolio_monitor()
        perf_data = performance_tracker.fetch_benchmark_performance()
        rel_data = performance_tracker.fetch_position_relative_performance(perf_data)
        spot_data = performance_tracker.fetch_spot_prices(perf_data)
        port_data = performance_tracker.fetch_portfolio_performance()
        sector_data = sector_monitor.fetch_sector_data()

        s3_text = _format_section3_portfolio_status(
            monitor_data, perf_data, rel_data, spot_data, port_data, sector_data
        )
        alert_tickers = [
            t for t, d in monitor_data.get("positions", {}).items()
            if d.get("move_flag") in ("LARGE_UP", "LARGE_DOWN")
        ]
        s3_kb = make_tickers_keyboard(alert_tickers) if alert_tickers else None
        sections.append((s3_text, s3_kb))
        all_text.append(s3_text)
    except Exception as exc:
        logger.error("Section 3 (Portfolio Status) FAILED: %s", exc, exc_info=True)
        sections.append((f"⚠️ Portfolio status unavailable: {exc}", None))

    # ── SECTION 4: NEWS & SIGNALS ────────────────────────────────
    # Consolidates: news triage + screener.
    try:
        triage = news_triage.run_news_triage(positions.get_all_tickers())
        df_screener = screener.run_screener()

        s4_text = _format_section4_news_signals(triage, df_screener)

        thesis_alert_tickers = [t for t, d in triage.items() if d.get("has_thesis_alert")]
        thesis_articles = [
            {"url": h.get("url", ""), "ticker": t}
            for t, d in triage.items()
            for h in d.get("headlines", [])
            if h.get("thesis_breaker_match")
        ]
        all_news_tickers = [t for t, d in triage.items() if d.get("headlines")]
        screener_tickers = (
            df_screener["ticker"].tolist()[:6] if df_screener is not None and len(df_screener) > 0 else []
        )

        article_kb = make_article_keyboard(thesis_articles) if thesis_articles else None
        news_nav_kb = make_news_actions(thesis_alert_tickers, all_news_tickers)
        screener_kb = make_tickers_keyboard(screener_tickers) if screener_tickers else None

        sections.append((s4_text, article_kb))
        if news_nav_kb:
            sections.append(("", news_nav_kb))
        if screener_kb:
            sections.append(("", screener_kb))
        all_text.append(s4_text)
    except Exception as exc:
        logger.error("Section 4 (News & Signals) FAILED: %s", exc, exc_info=True)
        sections.append((f"⚠️ News & signals unavailable: {exc}", None))

    # ── SECTION 5: MORNING SYNTHESIS ────────────────────────────
    # The only AI synthesis call in the brief — see module docstring for why
    # the per-section calls were removed. Sees every section's text, so it's
    # the one place cross-section connections can be made explicit.
    try:
        monitoring_items = load_monitoring()
        full_synth = synthesize_full_brief(
            all_sections_text="\n\n---SECTION BREAK---\n\n".join(all_text),
            regime_flags=regime_flags,
            monitoring_items=monitoring_items,
            max_tokens=1800,  # raised from 1200 — that still truncated on real brief data
        )
        sections.append((
            f"{_DIVIDER_HEAVY}\n🎯 MORNING SYNTHESIS\n{_DIVIDER_HEAVY}\n{full_synth}",
            make_main_menu(),
        ))
        _parse_and_persist_monitoring(full_synth, source="full_brief")
    except Exception as exc:
        logger.error("Section 5 (Morning Synthesis) FAILED: %s", exc, exc_info=True)
        sections.append((f"⚠️ Morning synthesis unavailable: {exc}", make_main_menu()))

    # Footer
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    sections.append((f"Generated {ts}", None))

    return [s for s in sections if s[0] or s[1]]


def save_brief_to_thread(sections: list[tuple[str, "InlineKeyboardMarkup | None"]], thread_manager) -> None:
    """Persist the brief's synthesis sections to a persistent `topic_BRIEF` thread.

    This is what lets `Advisor._get_cross_thread_context()` (see
    equity.telegram.advisor) surface the morning brief in every
    conversation — ticker, macro, portfolio, general — not just a
    `/discuss` about the brief itself. Only the 🎯 full-brief synthesis is
    saved (there's no longer a per-section 💡 synthesis to also match — see
    module docstring); the raw data sections (screener output, eco
    calendar, etc.) are already available live elsewhere and would just
    bloat this thread's history.

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
