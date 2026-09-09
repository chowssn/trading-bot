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


def _extract_text(response, label: str) -> str:
    """Response text, warning if the model was cut off at max_tokens.

    A `max_tokens` stop is invisible in the returned text — it just ends
    mid-sentence — but it silently drops whatever the prompt asked for
    last. For the two synthesis calls that request a NEW MONITORING ITEMS
    block (which the prompts place at the end), that meant the block was
    never emitted and `_parse_and_persist_monitoring()` found nothing to
    parse. This was the actual cause of the empty monitoring list: the
    parser was looking for a section the response never got to.
    """
    if getattr(response, "stop_reason", None) == "max_tokens":
        logger.warning(
            "%s: response hit max_tokens and was truncated — trailing sections "
            "(including any NEW MONITORING ITEMS block) are missing. Raise max_tokens.",
            label,
        )
    return response.content[0].text.strip()


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
        result = _extract_text(r, f"Section synthesis ({section_name})")
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

<<<NEW_MONITORING_ITEMS>>>
List items in EXACTLY this format, one per line, pipe-delimited:
TICKER | specific measurable condition to watch | high/medium/low

Example:
TSLA | Megapack margin trajectory in next earnings | high
PLTR | RSI 14D — watch for oversold + turn signal as framework entry setup | medium
GOOGL | 200D MA test — follow-through in next 3 sessions determines if support holds | high

Rules:
- Use the exact pipe-delimited format above — no other format
- TICKER must be the exact ticker symbol (TSLA not Tesla)
- Condition must be specific and measurable
- Only include if genuinely worth tracking for 3+ sessions
- Maximum 3 items per synthesis call
- If nothing warrants monitoring, write: NONE
<<<END_MONITORING_ITEMS>>>

This delimited block is REQUIRED — always emit it, and emit it last. Keep
the preceding sections short enough that you always reach it.

Be direct. Be specific. Name tickers and levels. Avoid generic macro commentary
that does not connect to a specific position or actionable decision."""

    try:
        r = client.messages.create(
            model=MODEL,
            # Richest section, and the one whose prompt ends with the
            # required NEW MONITORING ITEMS block. At 800 the response was
            # reliably hitting max_tokens partway through the MONITORING
            # UPDATES section and never reaching that block, so monitoring
            # items were silently never captured. Sized with headroom over
            # the ~800-token bodies actually observed.
            max_tokens=1400,
            messages=[{"role": "user", "content": prompt}],
        )
        result = _extract_text(r, "Performance synthesis")
        _save_cache(path, result)
        return result
    except Exception as exc:  # anthropic SDK can raise a variety of API/network errors
        logger.warning("Performance synthesis failed: %s", exc)
        return f"[Performance synthesis unavailable: {exc}]"


def synthesize_global_signals(
    section_data: str,
    regime_flags: list[str] | None = None,
    monitoring_items: list[dict] | None = None,
) -> str:
    """Dedicated synthesis for the global signals section — same
    monitoring-list-integration and NEW MONITORING ITEMS pattern as
    `synthesize_performance()`, since this section spans multiple asset
    classes simultaneously and its cross-asset reads are exactly the kind
    of thing worth tracking across sessions (see `_parse_and_persist_monitoring()`).

    Higher max_tokens than `synthesize_section()`'s generic per-section
    call — cross-asset synthesis has more to potentially cover (futures,
    vol, crypto, international, credit, ratios) than a single-asset-class
    section. Cached `SYNTHESIS_CACHE_HOURS` hours by content hash.
    """
    data_hash = hashlib.md5(section_data.encode()).hexdigest()
    path = _cache_path("global_signals", data_hash)
    cached = _load_cache(path)
    if cached:
        return cached

    regime_str = ", ".join(regime_flags) if regime_flags else "No active regime flags"
    positions_ctx = _get_positions_context()

    monitoring_str = ""
    if monitoring_items:
        monitoring_str = "\n\nACTIVE MONITORING LIST (carry these forward unless dismissed):\n"
        for item in monitoring_items:
            monitoring_str += (
                f'- [{item["ticker"]}] {item["item"]} '
                f'(priority: {item.get("priority", "medium")}, age: {item.get("age_days", 0)}d)\n'
            )

    prompt = f"""{FRAMEWORK}

Current regime: {regime_str}

{positions_ctx}
{monitoring_str}

GLOBAL SIGNALS DATA:
{section_data}

You are writing the global signals synthesis for a discretionary portfolio manager.
This section covers futures, volatility, crypto, international indices, credit proxies,
and cross-asset ratios. It should inform real-time decisions and regime assessment.

Length guidance: Scale to signal density. A quiet session with no notable moves
warrants 100-150 words. A session with multiple cross-asset signals or regime-relevant
moves warrants 200-350 words. Never pad — every sentence must earn its place.

Structure as follows:

SUMMARY (2-4 sentences):
What is the dominant cross-asset theme right now?
Be specific about which instruments are confirming vs. contradicting each other.
Identify whether the signal pattern is risk-on, risk-off, growth-driven, inflation-driven,
liquidity-driven, or idiosyncratic. Name specific levels and moves.
Example of good summary: "Equity futures flat but VIX/VVIX diverging upward while
copper/gold ratio falls — surface calm masking growing hedging demand. International
indices bifurcating: Asia +2% while Europe flat, suggesting regional rather than global
risk-on. Credit spreads (HYG -0.3%) contradicting equity futures strength."
Example of bad summary: "Markets showing mixed signals across asset classes today."

PORTFOLIO IMPLICATIONS (3-5 bullets, scaled to signal significance):
- For each cross-asset signal: name the specific portfolio positions affected and how.
  Be direct — "KOSPI +3% confirms TSM thesis" not "Korean markets moving positively."
- For volatility signals: if VIX/VVIX elevated, name which positions face multiple
  compression risk and whether the QQQ/SPY put hedge overlay is relevant.
- For FX moves: name which positions are directly affected
  (USD/JPY → SMFG thesis; USD/CNH → BYDDY, TSM, EWW; copper → FCX, CAT, industrials).
- For yield moves: name duration-sensitive positions (TLT thesis, MSFT/AMZN/GOOGL
  multiple compression risk) and the bp move relative to thesis-breaker levels.
- For crypto: flag if BTC move is risk-appetite signal relevant to broader positioning.
- For credit proxies: HYG vs LQD spread changes indicate credit stress — name which
  speculative positions (PLTR, RDDT, UMAC, QBTS) are most credit-sensitive.

CROSS-ASSET VERDICT (1-2 sentences):
What does the aggregate signal say about the current regime?
Is this a CONFIRM (signals consistent with existing regime), CONTRADICT (signals
inconsistent — warrants reassessment), or MIXED (no clear directional read)?
Name the single most important cross-asset signal today.

MONITORING UPDATES (only if monitoring list is non-empty):
For each active monitoring item related to macro/global signals:
one line stating current status. Escalate if worsening. Flag if resolved.

SUGGESTIONS (1-3 items):
Specific and actionable. Name the ticker and the exact condition.
For monitoring items: state what would trigger dismissal vs. escalation.
Never suggest adding to a position not in the dislocation zone.
If no action is warranted, say so in one sentence.

<<<NEW_MONITORING_ITEMS>>>
List items in EXACTLY this format, one per line, pipe-delimited:
TICKER | specific measurable condition to watch | high/medium/low

Example:
USDJPY | BOJ intervention risk approaching 160 — watch SMFG thesis | high
COPPER | Copper/gold ratio falling — growth concern signal, watch FCX thesis | medium
VVIX | VVIX trending toward 90 — tail risk building, watch speculative position sizes | high

Rules:
- Use the exact pipe-delimited format above — no other format
- TICKER must be the exact ticker/pair symbol (USDJPY not "the yen")
- Condition must be specific and measurable
- Only include if genuinely worth tracking for 3+ sessions
- Maximum 3 items per synthesis call
- If nothing warrants monitoring, write: NONE
<<<END_MONITORING_ITEMS>>>

This delimited block is REQUIRED — always emit it, and emit it last. Keep
the preceding sections short enough that you always reach it.

Be specific about cross-asset relationships. Name tickers. Name levels.
The person making decisions needs to know WHAT to do, not just WHAT is happening."""

    try:
        r = client.messages.create(
            model=MODEL,
            # Same truncation problem as synthesize_performance() at 700 —
            # the trailing NEW MONITORING ITEMS block was being cut off.
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}],
        )
        result = _extract_text(r, "Global signals synthesis")
        _save_cache(path, result)
        return result
    except Exception as exc:  # anthropic SDK can raise a variety of API/network errors
        logger.warning("Global signals synthesis failed: %s", exc)
        return f"[Global signals synthesis unavailable: {exc}]"


# Sentinel the synthesis prompts now ask for. Unlike a prose header it
# can't collide with the model merely *mentioning* the section by name in
# a preamble, and its explicit end marker means a blank line between items
# no longer truncates the block.
MONITORING_BLOCK_START = "<<<NEW_MONITORING_ITEMS>>>"
MONITORING_BLOCK_END = "<<<END_MONITORING_ITEMS>>>"

# Where a legacy (undelimited) block ends: two blank lines, a markdown
# horizontal rule ON ITS OWN LINE, or end of text. The rule alternative is
# anchored to a full line on purpose — an unanchored `---` also matched the
# `|---|---|` separator row inside a markdown table, cutting the block off
# before any of the table's item rows.
_BLOCK_END = r"(?:\n\s*\n\s*\n|\n\s*-{3,}\s*(?:\n|$)|\Z)"

# Tried in order, but see `_extract_monitoring_block()`: the FIRST pattern
# to match is not automatically the winner. Every pattern is tried at every
# position and the candidate yielding the most parseable item lines wins.
#
# That matters because these patterns overlap. "NEW MONITORING ITEMS" is a
# substring of "**NEW MONITORING ITEMS**", and the plain-header pattern also
# matches where the model merely *names* the section in a preamble ("I'll
# close with NEW MONITORING ITEMS and SUGGESTIONS") — under first-match-wins
# either case captured an empty or junk block and the real items, sitting a
# few lines further down, were never looked for.
_MONITORING_HEADER_PATTERNS = [
    # Explicit delimiter pair — most reliable, no ambiguity about where the
    # block ends.
    rf"{re.escape(MONITORING_BLOCK_START)}(.*?){re.escape(MONITORING_BLOCK_END)}",
    # Same delimiter, but the response was cut off before the end marker.
    rf"{re.escape(MONITORING_BLOCK_START)}(.*)\Z",
    # Legacy prose headers, kept so a cached synthesis from before the
    # delimiter change still parses.
    r"\*\*NEW MONITORING ITEMS\*\*[:\s]*(.*?)" + _BLOCK_END,
    r"NEW MONITORING ITEMS[:\s]*(.*?)" + _BLOCK_END,
    r"NEW ITEMS TO MONITOR[:\s]*(.*?)" + _BLOCK_END,
    r"MONITORING ITEMS[:\s]*(.*?)" + _BLOCK_END,
]


_NO_ITEMS_MARKERS = ("none", "(none)", "n/a", "-")

# Column labels from a markdown table header row. The model sometimes
# renders the items as a table instead of bare pipe-delimited lines, and
# its header row is pipe-delimited too — without this it parsed into a
# monitoring item for a ticker literally named "TICKER".
_TABLE_HEADER_TOKENS = {"ticker", "symbol", "item", "condition", "priority", "name", "level"}


def _emitted_empty_monitoring_block(synthesis_text: str) -> bool:
    """True if the response emitted the monitoring block but declared no items.

    Distinguishes "the model got to the block and said NONE" (expected,
    quiet) from "the block never appeared at all" (a real failure worth a
    warning — a truncated response, or a header shape nothing matches).
    """
    start = synthesis_text.find(MONITORING_BLOCK_START)
    if start == -1:
        return False
    body = synthesis_text[start + len(MONITORING_BLOCK_START):]
    end = body.find(MONITORING_BLOCK_END)
    if end != -1:
        body = body[:end]
    lines = [ln.strip().lstrip("- •*").strip().lower() for ln in body.split("\n")]
    content = [ln for ln in lines if ln]
    return all(ln in _NO_ITEMS_MARKERS for ln in content)


def _count_item_lines(block: str) -> int:
    """How many lines in `block` look like `TICKER | condition | priority`."""
    return sum(1 for line in block.split("\n") if "|" in line.strip().lstrip("- •*").strip())


def _extract_monitoring_block(synthesis_text: str) -> tuple[str | None, str | None]:
    """Best monitoring block in `synthesis_text`, as `(block, pattern_used)`.

    Scores every match of every pattern by how many pipe-delimited item
    lines it contains and returns the highest scorer, so a pattern that
    matches early but captures nothing useful can't shadow one that
    actually found the items. Ties break toward the earlier pattern in
    `_MONITORING_HEADER_PATTERNS`, which is ordered most- to least-specific.
    """
    best_block = None
    best_pattern = None
    best_score = 0

    for pattern in _MONITORING_HEADER_PATTERNS:
        for match in re.finditer(pattern, synthesis_text, re.DOTALL | re.IGNORECASE):
            block = match.group(1).strip()
            if not block:
                continue
            score = _count_item_lines(block)
            if score > best_score:
                best_block, best_pattern, best_score = block, pattern, score

    return best_block, best_pattern


def _parse_and_persist_monitoring(synthesis_text: str, source: str = "synthesis") -> None:
    """Extracts the NEW MONITORING ITEMS block from a synthesis call's
    output and persists it via `equity.data.monitoring.add_monitoring_items()`.

    Only the strict `TICKER | condition | priority` line format is parsed
    — both synthesis prompts now say "use EXACTLY this format" for exactly
    this reason. An earlier version of this function also tried to recover
    a ticker from colon/dash-separated prose lines when a line had no `|`;
    dropped it — scoped to a block that's supposed to be item lines, a
    stray line of ordinary prose (e.g. "(none)" written as a full sentence,
    or the model wandering back into commentary before the actual `\n\n`
    break) could still parse as "TICKER: the rest of the sentence" and
    create a bogus monitoring item, which is worse than just not parsing
    that line.

    `source` is which synthesis call this came from (e.g.
    "performance_synthesis", "global_signals_synthesis") — stored per item
    so `equity.data.monitoring`'s history shows where it originated.

    Never raises — a malformed or absent block just means nothing new gets
    persisted this run, which is no worse than the monitoring list not
    growing that day. Logs at each stage (which header pattern matched, if
    any; how many items parsed) so an actual failure — as opposed to the
    model legitimately writing "(none)" for the block — is diagnosable
    from the logs rather than just silently not showing up.
    """
    if not synthesis_text:
        logger.debug("_parse_and_persist_monitoring: empty synthesis text (source=%s)", source)
        return

    logger.debug(
        "_parse_and_persist_monitoring: parsing %d chars from %s",
        len(synthesis_text), source,
    )

    block, pattern = _extract_monitoring_block(synthesis_text)

    if not block and _emitted_empty_monitoring_block(synthesis_text):
        # The model reached the block and correctly reported nothing to
        # track. That's a normal outcome, not a parse failure — don't warn.
        logger.debug(
            "_parse_and_persist_monitoring: model reported no items to monitor (source=%s)", source,
        )
        return

    if not block:
        # WARNING, not DEBUG, and with a much longer tail: when this fires
        # the items are silently lost, and the tail is the only evidence of
        # why. A tail that stops mid-sentence means the response hit
        # max_tokens before it ever reached the monitoring block — that is a
        # token-budget problem, not a regex problem, and no additional
        # pattern will fix it (see the max_tokens note on the synthesis
        # calls above).
        logger.warning(
            "_parse_and_persist_monitoring: no monitoring block found in %d char synthesis "
            "(source=%s). Last 500 chars: %r",
            len(synthesis_text), source, synthesis_text[-500:],
        )
        return

    logger.debug(
        "_parse_and_persist_monitoring: matched pattern %r -> %d char block",
        pattern[:50], len(block),
    )

    new_items = []
    for line in block.split("\n"):
        line = line.strip().lstrip("- •*").strip()
        if "|" not in line:
            continue
        # Markdown table separator rows ("|---|---|") aren't items.
        if set(line) <= set("|- :"):
            continue
        # A markdown table row ("| TSLA | cond | high |") splits into empty
        # leading/trailing fields — drop them so the ticker lands in parts[0].
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) < 2:
            continue
        ticker = parts[0].upper().strip("*_")
        item_text = parts[1]
        if ticker.lower() in _TABLE_HEADER_TOKENS:
            continue
        priority = parts[2].lower().strip("*_ ") if len(parts) > 2 else "medium"
        priority = priority if priority in ("high", "medium", "low") else "medium"
        if ticker and item_text:
            new_items.append({
                "ticker": ticker,
                "item": item_text,
                "priority": priority,
                "source": source,
            })

    if not new_items:
        logger.debug(
            "_parse_and_persist_monitoring: header matched but no valid items parsed "
            "(source=%s). Block: %r",
            source, block[:200],
        )
        return

    logger.info(
        "_parse_and_persist_monitoring: adding %d item(s) from %s: %s",
        len(new_items), source, [i["ticker"] for i in new_items],
    )
    try:
        from equity.data.monitoring import add_monitoring_items

        add_monitoring_items(new_items)
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

When assessing these, consider:
- Cross-asset confirmation/contradiction: do futures, vol, credit, and international indices tell the same story or different stories?
- If VIX/VVIX elevated: which speculative positions face the most compression risk?
- If yield moves are significant: TLT thesis status and duration exposure across the book.
- If FX moves are significant: SMFG (JPY), BYDDY/EWW/TSM (CNH), FCX/commodities (DXY).
- If copper/gold ratio moving: FCX, CAT, industrials thesis implications.
- If crypto moving significantly: risk appetite signal relevant to RDDT, PLTR positioning.
The most actionable implications often come from CONTRADICTIONS between asset classes, not confirmations. Flag these explicitly when present.

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
        result = _extract_text(r, "Full brief synthesis")
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

    global_test_data = "ES=F +0.1% | VIX 14.5 | VVIX 84.4 | BTC-USD +0.4% | Nikkei 225 +2.0% | HYG -0.06%"
    global_result = synthesize_global_signals(global_test_data, ["DOLLAR_STRENGTH"])
    print()
    print("=== Global signals synthesis test ===")
    print(global_result)
