"""AI-powered synthesis for each morning brief section, plus an end-of-brief summary.

Every call to `synthesize_section()`/`synthesize_full_brief()` is cached to
`equity/data/cache/synthesis/` for `SYNTHESIS_CACHE_HOURS` hours, keyed by
an MD5 hash of the section's (already-formatted) text plus the calendar
date — so re-running the brief mid-morning (e.g. after a `/brief` retry)
never burns a second Claude call over identical data, but a materially
different snapshot (a new hash) always gets its own synthesis.

Both functions never raise: any Anthropic API failure is caught and
returned as a `'[... synthesis unavailable: ...]'` string rather than
propagating, so one bad synthesis call never takes down the rest of the
brief (same philosophy as every `brief_builder.py` section).
"""

import hashlib
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Same model as equity.telegram.advisor's Advisor — not imported from there
# to keep this module's only dependency on the advisor package being the
# `anthropic` SDK itself, not the (heavier) Advisor/ThreadManager import
# chain.
MODEL = "claude-sonnet-4-6"

SYNTHESIS_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache" / "synthesis"
SYNTHESIS_CACHE_HOURS = 6

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def _cleanup_old_synthesis_cache() -> None:
    """Delete synthesis cache files from prior days on module load.

    Cache entries are never read back past today (see `_cache_path`/
    `_load_cache`), so anything left over from an earlier day is dead
    weight — this just keeps the cache dir from growing unbounded.
    """
    if not SYNTHESIS_CACHE_DIR.exists():
        return
    today_str = datetime.now().strftime("%Y-%m-%d")
    for f in SYNTHESIS_CACHE_DIR.glob("*.json"):
        if today_str not in f.name:
            try:
                f.unlink()
            except OSError as exc:
                logger.warning("Failed to remove stale synthesis cache %s: %s", f, exc)


_cleanup_old_synthesis_cache()

_POSITIONS_CONTEXT_CACHE: str | None = None


def _get_positions_context() -> str:
    """Compact positions/watchlist summary, built once per process and reused across calls."""
    global _POSITIONS_CONTEXT_CACHE
    if _POSITIONS_CONTEXT_CACHE is None:
        from equity.config.positions import POSITIONS, WATCHLIST

        lines = ["Current positions and thesis:"]
        for ticker, pos in POSITIONS.items():
            thesis_short = pos.get("thesis", "")[:120]
            breakers = pos.get("thesis_breakers", [])[:2]
            lines.append(f'  {ticker} ({pos.get("tier", "")}, {pos.get("sector", "")}): {thesis_short}')
            if breakers:
                lines.append(f'    Thesis-breakers: {"; ".join(breakers)}')
        lines.append("Watchlist: " + ", ".join(WATCHLIST.keys()))
        _POSITIONS_CONTEXT_CACHE = "\n".join(lines)
    return _POSITIONS_CONTEXT_CACHE


def _cache_path(section_name: str, data_hash: str) -> Path:
    SYNTHESIS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    return SYNTHESIS_CACHE_DIR / f"{section_name}_{date_str}_{data_hash[:8]}.json"


def _load_cache(path: Path) -> str | None:
    if not path.exists():
        return None
    # Belt-and-suspenders: _cache_path() already bakes today's date into
    # the filename, so this should never trip. Guards against a future
    # refactor decoupling path construction from _load_cache.
    today_str = datetime.now().strftime("%Y-%m-%d")
    if today_str not in path.name:
        return None
    try:
        with open(path) as f:
            cached = json.load(f)
        if (datetime.now().timestamp() - cached["ts"]) / 3600 < SYNTHESIS_CACHE_HOURS:
            return cached["text"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning("Synthesis cache at %s unreadable, ignoring: %s", path, exc)
    return None


def _save_cache(path: Path, text: str) -> None:
    try:
        with open(path, "w") as f:
            json.dump({"text": text, "ts": datetime.now().timestamp()}, f)
    except OSError as exc:
        logger.warning("Failed to write synthesis cache to %s: %s", path, exc)


SECTION_FOCUS = {
    "market_snapshot": "market regime, risk appetite, and cross-asset signals",
    "rates":           "yield curve shape, rate trajectory, and duration risk for equity positions",
    "fx":              "USD direction, carry implications, and exposure for international holdings",
    "commodities":     "commodity cycle signals and implications for energy/materials positions",
    # "performance" intentionally has no entry here — it's synthesized by
    # the dedicated synthesize_performance() below instead of the generic
    # synthesize_section() path, so this map never needs to describe it.
    "news":            "thesis-relevant developments, thesis-breaker risk, and urgency assessment",
    "sector":          "concentration risk, correlation changes, and options positioning signals",
    "earnings":        "upcoming earnings risk and pre-earnings positioning implications",
}

FRAMEWORK = """Investment framework: Quality-at-discount, macro-aware, fundamentals-first.
Goal: active portfolio management with consolidated global market view.
Four-question framework: (1) Great business? ROIC-led. (2) Market mispricing? 1Y return + RSI dislocation.
(3) Why might market be right? Stress-test the thesis. (4) Good entry? RSI 14D turning.
Exit rule: thesis broken — not price target. Build positions in pieces.
Prefer smaller frequent wins over large infrequent wins with deep drawdowns."""


def synthesize_section(section_name: str, section_data: str, regime_flags: list[str] | None = None) -> str:
    """Concise synthesis for one brief section. Cached `SYNTHESIS_CACHE_HOURS` hours by content hash."""
    data_hash = hashlib.md5(section_data.encode()).hexdigest()
    path = _cache_path(section_name, data_hash)
    cached = _load_cache(path)
    if cached:
        return cached

    regime_str = ", ".join(regime_flags) if regime_flags else "No active regime flags"
    focus = SECTION_FOCUS.get(section_name, "key signals and portfolio implications")
    positions_ctx = _get_positions_context()

    prompt = f"""{FRAMEWORK}

Current regime: {regime_str}

{positions_ctx}

{section_name.upper()} DATA:
{section_data}

Provide a concise synthesis focused on {focus}. Structure as:

SUMMARY (2-3 sentences max): The single most important signal in this data right now.

PORTFOLIO IMPLICATIONS (2-3 bullets max): Direct, material implications for current positions or watchlist only. Name the ticker. Skip if no direct implication.

SUGGESTIONS (1-2 bullets max): The most plausible actionable suggestion given this data and regime. Name the ticker and action if relevant. If nothing is clearly actionable, say so.

IMPORTANT constraints for SUGGESTIONS:
- For existing positions already showing significant unrealized gains (>20%), do NOT suggest adding. Suggest hold/trim assessment instead.
- For positions at 52W or multi-year highs, flag the entry risk explicitly rather than suggesting addition.
- Only suggest adding to a position if it is in the dislocation zone (down 10-50% from highs) AND RSI is confirming a turn.
- Suggestions should be specific but acknowledge uncertainty — avoid language like "add on this session" which implies high confidence.
- If a suggestion conflicts with the four-question framework (e.g. suggesting adding to something not in the dislocation zone), flag the conflict rather than making the suggestion.

Be direct and specific. Avoid generic observations. If a signal is ambiguous, say so.
Under 200 words total."""

    try:
        r = client.messages.create(model=MODEL, max_tokens=400, messages=[{"role": "user", "content": prompt}])
        result = r.content[0].text.strip()
        _save_cache(path, result)
        return result
    except Exception as exc:  # anthropic SDK can raise a variety of API/network errors
        logger.warning("Section synthesis failed for %s: %s", section_name, exc)
        return f"[Synthesis unavailable: {exc}]"


def synthesize_performance(
    section_data: str,
    regime_flags: list[str] | None = None,
    monitoring_items: list[dict] | None = None,
    cross_thread_context: str = "",
) -> str:
    """Dedicated performance-section synthesis, richer than `synthesize_section()`.

    The performance section carries the full day's price action, so it
    gets its own prompt rather than the generic per-section one: explicit
    check-ins on the persistent monitoring list (see `equity.data.monitoring`),
    cross-section context from earlier brief sections, and a NEW MONITORING
    ITEMS block that `_parse_and_persist_monitoring()` parses back out.

    Cached `SYNTHESIS_CACHE_HOURS` hours, keyed by a hash of `section_data`
    + `cross_thread_context` — the monitoring list itself doesn't affect
    the cache key, so a monitoring item dismissed mid-cache-window doesn't
    force a fresh (and identical) Claude call.
    """
    data_hash = hashlib.md5((section_data + cross_thread_context).encode()).hexdigest()
    path = _cache_path("performance", data_hash)
    cached = _load_cache(path)
    if cached:
        return cached

    regime_str = ", ".join(regime_flags) if regime_flags else "No active regime flags"
    positions_ctx = _get_positions_context()

    monitoring_str = ""
    if monitoring_items:
        monitoring_str = "\n\nACTIVE MONITORING LIST (carry these forward unless dismissed):\n"
        for item in monitoring_items:
            age_days = item.get("age_days", 0)
            monitoring_str += (
                f'- [{item["ticker"]}] {item["item"]} '
                f'(added {age_days}d ago, priority: {item.get("priority", "medium")})\n'
            )

    prompt = f"""{FRAMEWORK}

Current regime: {regime_str}

{positions_ctx}

CROSS-SECTION CONTEXT (from today's other brief sections and recent threads):
{cross_thread_context[:1000] if cross_thread_context else "Not available"}
{monitoring_str}

PERFORMANCE DATA:
{section_data}

You are writing the performance synthesis for a discretionary portfolio manager's morning brief.
This section should be the most actionable part of the brief — it has the full day's price action
and should integrate context from the monitoring list and cross-section data.

Length guidance: Scale to the day's significance. A quiet day warrants 100-150 words.
A day with significant divergences, thesis-proximity events, or monitoring list developments
warrants 200-300 words. Never pad — every sentence must earn its place.

Structure as follows:

SUMMARY (2-4 sentences):
Identify the dominant intraday theme across the portfolio. Go beyond surface labels —
not just "idiosyncratic selling" but what KIND: bifurcation within a sector (winners vs losers
on the same catalyst), multiple compression at elevated RSI, profit-taking after extended run,
news-driven vs tape-driven, etc. Name the specific positions driving the theme.

PORTFOLIO IMPLICATIONS (3-5 bullets, scaled to the day):
- For each materially moving position: name the specific divergence vs sector, classify the move
  type, and state whether it is thesis-proximity (close to a breaker), thesis-neutral (noise),
  or thesis-constructive (thesis delivering).
- For positions on the monitoring list: explicitly check in on each item. Has the condition
  improved, worsened, or stayed the same? Update the status.
- For positions approaching technical levels (RSI extremes, MA crosses, 52W high/low proximity):
  name the level and its significance to the entry/exit framework.
- Flag any earnings or catalyst dates within 14 days for held positions — these are
  thesis-confirmation or thesis-breaker events.
- If sector divergence exceeds 3% for any position, classify as WATCH or FLAG explicitly.

MONITORING UPDATES (only if monitoring list is non-empty):
For each active monitoring item: one line stating current status vs. when it was added.
If a condition has resolved (positively or negatively), say so clearly.
If it has worsened, escalate priority.

SUGGESTIONS (1-3 items, scaled to the day):
Be specific about what would constitute a framework-compliant setup.
For WATCH items: name the exact RSI level, price level, or event that would trigger action.
For monitoring items: state what would cause dismissal vs. escalation.
If no action is warranted, say so in one sentence — do not invent suggestions.
Never suggest adding to a position that is NOT in the dislocation zone
(down 10-50% from highs with RSI confirming a turn) unless explicitly noted as an exception.

NEW MONITORING ITEMS:
List any new items that emerged from today's data that should be tracked going forward.
Format: TICKER | specific condition to watch | priority (high/medium/low)
Only add items that are genuinely worth tracking for multiple days.
Examples of good monitoring items:
  TSLA | Megapack margin trajectory in next earnings | high
  TSLA | RSI 14D — watch for oversold + turn signal as framework entry setup | medium
  PLTR | Multi-year high resistance — do not add until RSI confirms pullback absorbed | medium
  GOOGL | 200D MA test — follow-through in next 3 sessions determines if support holds | high
Examples of bad monitoring items (too vague, not actionable):
  MSFT | watch for weakness | low
  General | market conditions | medium

Be direct. Be specific. Name tickers and levels. Avoid generic macro commentary
that does not connect to a specific position or actionable decision."""

    try:
        r = client.messages.create(
            model=MODEL,
            max_tokens=800,  # higher limit than synthesize_section() — this is the richest section
            messages=[{"role": "user", "content": prompt}],
        )
        result = r.content[0].text.strip()
        _save_cache(path, result)
        return result
    except Exception as exc:  # anthropic SDK can raise a variety of API/network errors
        logger.warning("Performance synthesis failed: %s", exc)
        return f"[Performance synthesis unavailable: {exc}]"


def _parse_and_persist_monitoring(synthesis_text: str) -> None:
    """Extracts the NEW MONITORING ITEMS block from `synthesize_performance()`'s
    output and persists it via `equity.data.monitoring.add_monitoring_items()`.

    Expected line format: `TICKER | condition | priority`. Never raises —
    a malformed or absent block just means nothing new gets persisted this
    run, which is no worse than the monitoring list not growing that day.
    """
    match = re.search(
        r"NEW MONITORING ITEMS[:\s]*(.*?)(?:\n\n|\Z)",
        synthesis_text, re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return

    block = match.group(1).strip()
    new_items = []

    for line in block.split("\n"):
        line = line.strip().lstrip("- •").strip()
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 2:
            continue
        ticker = parts[0].upper()
        item_text = parts[1]
        priority = parts[2].lower() if len(parts) > 2 else "medium"
        priority = priority if priority in ("high", "medium", "low") else "medium"
        if ticker and item_text:
            new_items.append({
                "ticker": ticker,
                "item": item_text,
                "priority": priority,
                "source": "performance_synthesis",
            })

    if new_items:
        try:
            from equity.data.monitoring import add_monitoring_items

            add_monitoring_items(new_items)
            logger.info("_parse_and_persist_monitoring: added %d items", len(new_items))
        except Exception as exc:
            logger.warning("_parse_and_persist_monitoring: failed to persist items: %s", exc)


def synthesize_full_brief(all_sections_text: str, regime_flags: list[str] | None = None) -> str:
    """End-of-brief synthesis covering the full morning data. Cached `SYNTHESIS_CACHE_HOURS` hours by content hash."""
    data_hash = hashlib.md5(all_sections_text.encode()).hexdigest()
    path = _cache_path("full_brief", data_hash)
    cached = _load_cache(path)
    if cached:
        return cached

    regime_str = ", ".join(regime_flags) if regime_flags else "No active regime flags"
    positions_ctx = _get_positions_context()

    prompt = f"""{FRAMEWORK}

Current regime: {regime_str}

{positions_ctx}

FULL BRIEF DATA (truncated to 3000 chars):
{all_sections_text[:3000]}

Provide a closing synthesis of the full morning brief. Structure as:

OVERALL ASSESSMENT (2-3 sentences): The single most important thing to know this morning. The dominant theme across all sections.

TOP 3 PORTFOLIO IMPLICATIONS (3 bullets max): The most material cross-section implications for current positions. Prioritize by urgency and magnitude. Name the ticker.

TODAY'S FOCUS (1-2 bullets): If you had to focus on one position decision and one thing to monitor today — what are they? Be specific.

IMPORTANT constraints for the position decision in TODAY'S FOCUS:
- For existing positions already showing significant unrealized gains (>20%), do NOT suggest adding. Suggest hold/trim assessment instead.
- For positions at 52W or multi-year highs, flag the entry risk explicitly rather than suggesting addition.
- Only suggest adding to a position if it is in the dislocation zone (down 10-50% from highs) AND RSI is confirming a turn.
- Be specific but acknowledge uncertainty — avoid language like "add on this session" which implies high confidence.
- If the decision conflicts with the four-question framework (e.g. suggesting adding to something not in the dislocation zone), flag the conflict rather than making the suggestion.

RISK TO WATCH (1 sentence): The single biggest risk to the current portfolio surfaced by this morning's data.

Direct, specific, prioritized. Under 250 words."""

    try:
        r = client.messages.create(model=MODEL, max_tokens=500, messages=[{"role": "user", "content": prompt}])
        result = r.content[0].text.strip()
        _save_cache(path, result)
        return result
    except Exception as exc:  # anthropic SDK can raise a variety of API/network errors
        logger.warning("Full-brief synthesis failed: %s", exc)
        return f"[Full brief synthesis unavailable: {exc}]"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    test_data = "SPY -0.2% | QQQ -0.6% | VIX 14.4 | 10Y 4.72% (+5bp) | Regime: DOLLAR_STRENGTH"
    result = synthesize_section("market_snapshot", test_data, ["DOLLAR_STRENGTH", "RATES_RISING"])
    print("=== Section synthesis test ===")
    print(result)
    print()
    full = synthesize_full_brief(test_data, ["DOLLAR_STRENGTH"])
    print("=== Full brief synthesis test ===")
    print(full)
