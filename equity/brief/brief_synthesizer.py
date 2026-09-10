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

# Per-section token budget for the 5-section brief (see
# equity/brief/brief_builder.py's build_morning_brief()). Maxima, not
# targets — the dense SIGNAL/POSITIONS AFFECTED/ACTION format synthesize_section()
# and synthesize_full_brief() now use rarely needs the full budget. Replaces the
# pre-reorg split (market_snapshot 400 + global_signals 1200 + performance 1400 +
# full_brief 500 =~ 3500, plus a second sector/news synthesize_section() call each
# at 400) with one call per section, totaling ~3000 tokens across a brief instead
# of ~6100.
SYNTHESIS_MAX_TOKENS = {
    "global_markets": 800,
    "portfolio_status": 900,
    "news_signals": 600,
    "full_brief": 700,
}

# Pre-reorg section names that used to be synthesized separately, keyed by
# the new 5-section name that now covers them. synthesize_section() checks
# these as a cache fallback so a synthesis cached under an old name earlier
# today still counts as a hit instead of forcing a redundant Claude call —
# see _load_cache_with_aliases().
SECTION_ALIASES = {
    "global_markets": ["market_snapshot", "global_signals"],
    "portfolio_status": ["performance", "sector"],
    "news_signals": ["news", "screener"],
}


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


def _load_cache_with_aliases(section_name: str, data_hash: str) -> str | None:
    """Today's synthesis cache for `section_name`, falling back to its
    pre-reorg alias names (`SECTION_ALIASES`) on a miss.

    The 5-section reorg means a section's formatted text (and so its
    content hash) rarely matches what used to be cached under the old,
    narrower section names — but when it does (e.g. a section whose
    upstream data hasn't changed since a pre-reorg cache entry was
    written earlier today), this counts it as a hit rather than paying
    for a redundant Claude call.
    """
    cached = _load_cache(_cache_path(section_name, data_hash))
    if cached:
        return cached
    for alias in SECTION_ALIASES.get(section_name, []):
        cached = _load_cache(_cache_path(alias, data_hash))
        if cached:
            return cached
    return None


def synthesize_section(
    section_name: str,
    section_data: str,
    regime_flags: list[str] | None = None,
    monitoring_items: list[dict] | None = None,
    max_tokens: int = 700,
) -> str:
    """Dense, structured synthesis for one of the 5 morning-brief sections.

    Structured density over narrative prose: SIGNAL / POSITIONS AFFECTED /
    ACTION, plus the same `<<<NEW_MONITORING_ITEMS>>>` block every other
    synthesis call emits (see `_parse_and_persist_monitoring()`) — every
    line must be specific enough to act on, nothing padded out to sound
    thorough. `max_tokens` is caller-controlled — see `SYNTHESIS_MAX_TOKENS`
    for the budget each of the 5 sections is called with.

    Cached `SYNTHESIS_CACHE_HOURS` hours by content hash; see
    `_load_cache_with_aliases()` for the pre-reorg cache fallback.
    """
    data_hash = hashlib.md5(section_data.encode()).hexdigest()
    path = _cache_path(section_name, data_hash)
    cached = _load_cache_with_aliases(section_name, data_hash)
    if cached:
        return cached

    regime_str = ", ".join(regime_flags) if regime_flags else "none"
    positions_ctx = _get_positions_context()

    monitoring_str = ""
    if monitoring_items:
        active = [i for i in monitoring_items if i.get("priority") in ("high", "medium")][:5]
        if active:
            monitoring_str = "\nACTIVE MONITORING:\n" + "\n".join(
                f'[{i["ticker"]}] {i["item"][:60]} ({i.get("age_days", 0)}d)' for i in active
            )

    prompt = f"""{FRAMEWORK}
Regime: {regime_str}

{positions_ctx}
{monitoring_str}

DATA:
{section_data[:2000]}

Respond in EXACTLY this format — no additions, no narrative prose:

SIGNAL: [1 sentence — most important thing right now, specific]

POSITIONS AFFECTED:
[TICKER] [↑/↓/→] [specific reason ≤10 words] [WATCH/ACT/HOLD]
(3-4 lines max — only material impacts, omit if none)

ACTION: [specific ticker + action] OR "no action warranted"
Condition: [what would change this]

<<<NEW_MONITORING_ITEMS>>> (use exactly this — include underscores)
[TICKER] | [specific measurable condition] | [high/medium/low]
(1-3 items max, or NONE)
<<<END_MONITORING_ITEMS>>> (use exactly this — include underscores)

Every line must be specific. No generic observations.
If unsure, write less not more."""

    try:
        r = client.messages.create(model=MODEL, max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}])
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
    _retry: bool = False,
) -> str:
    """Dense-format performance-section synthesis, same SIGNAL/POSITIONS
    AFFECTED/ACTION structure as `synthesize_section()` — no narrative
    prose. Still gets its own prompt (rather than the generic per-section
    one) for the monitoring-list check-ins and the NEW MONITORING ITEMS
    block that `_parse_and_persist_monitoring()` parses back out.

    `_retry=True` multiplies `SYNTHESIS_MAX_TOKENS['portfolio_status']` by
    1.3 — for a caller retrying after a response got cut off before reaching
    the monitoring-items block.

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
Regime: {regime_str}

{positions_ctx}
{monitoring_str}

PERFORMANCE DATA:
{section_data[:2000]}

Respond in EXACTLY this format — no markdown headers, no narrative prose:

SIGNAL: [1 sentence — dominant performance theme today]

POSITIONS AFFECTED:
[TICKER] [↑/↓/→] [vs sector divergence ≤10 words, include %] [WATCH/ACT/HOLD]
(4-5 lines — include any position with >2% sector divergence)

ACTION: [specific ticker + action] OR "no action warranted"
Condition: [what would change this]

MONITORING UPDATES:
[For each active monitoring item in the list: 1 line — status unchanged/improved/worsened]
(omit section entirely if no active monitoring items)

<<<NEW_MONITORING_ITEMS>>> (use exactly this — include underscores)
[TICKER] | [specific measurable condition] | [high/medium/low]
(1-3 items, or NONE)
<<<END_MONITORING_ITEMS>>> (use exactly this — include underscores)

Every line specific. No generic observations. No markdown."""

    # Use SYNTHESIS_MAX_TOKENS for consistency
    max_tokens_to_use = SYNTHESIS_MAX_TOKENS.get("portfolio_status", 900)
    if _retry:
        max_tokens_to_use = int(max_tokens_to_use * 1.3)

    try:
        r = client.messages.create(
            model=MODEL,
            max_tokens=max_tokens_to_use,
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
    override_max_tokens: int | None = None,
    _retry: bool = False,
) -> str:
    """Dense-format synthesis for the global signals section — same
    SIGNAL/POSITIONS AFFECTED/ACTION structure as `synthesize_section()`,
    no narrative prose. Still gets its own prompt (rather than the generic
    per-section one) for the monitoring-list integration and NEW MONITORING
    ITEMS block that `_parse_and_persist_monitoring()` parses back out,
    since this section spans multiple asset classes simultaneously and its
    cross-asset reads are exactly the kind of thing worth tracking across
    sessions.

    `override_max_tokens` bypasses `SYNTHESIS_MAX_TOKENS['global_markets']`
    for a caller with a different budget; `_retry=True` multiplies whichever
    of the two is in effect by 1.3 — for a caller retrying after a response
    got cut off before reaching the monitoring-items block. Cached
    `SYNTHESIS_CACHE_HOURS` hours by content hash.
    """
    data_hash = hashlib.md5(section_data.encode()).hexdigest()
    path = _cache_path("global_signals", data_hash)
    cached = _load_cache(path)
    if cached:
        return cached

    monitoring_str = ""
    if monitoring_items:
        monitoring_str = "\n\nACTIVE MONITORING LIST (carry these forward unless dismissed):\n"
        for item in monitoring_items:
            monitoring_str += (
                f'- [{item["ticker"]}] {item["item"]} '
                f'(priority: {item.get("priority", "medium")}, age: {item.get("age_days", 0)}d)\n'
            )

    prompt = f"""{FRAMEWORK}
Regime: {", ".join(regime_flags) if regime_flags else "none"}

{_get_positions_context()}
{monitoring_str}

GLOBAL SIGNALS DATA:
{section_data[:2000]}

Respond in EXACTLY this format — no markdown headers, no narrative prose:

SIGNAL: [1 sentence — dominant cross-asset theme, specific instruments named]

POSITIONS AFFECTED:
[TICKER] [↑/↓/→] [specific cross-asset reason ≤10 words] [WATCH/ACT/HOLD]
(3-4 lines max — FX/commodity/vol impacts on named positions only)

CROSS-ASSET: [CONFIRM/CONTRADICT/MIXED] — [1 sentence — what the contradiction or confirmation is]

ACTION: [specific action] OR "no action warranted"
Condition: [what would change this]

<<<NEW_MONITORING_ITEMS>>> (use exactly this — include underscores)
[TICKER] | [specific measurable condition] | [high/medium/low]
(1-3 items, or NONE)
<<<END_MONITORING_ITEMS>>> (use exactly this — include underscores)

Every line must be specific. No generic observations. No markdown formatting."""

    # Use SYNTHESIS_MAX_TOKENS for consistency
    max_tokens_to_use = override_max_tokens if override_max_tokens else SYNTHESIS_MAX_TOKENS.get("global_markets", 800)
    if _retry:
        max_tokens_to_use = int(max_tokens_to_use * 1.3)

    try:
        r = client.messages.create(
            model=MODEL,
            max_tokens=max_tokens_to_use,
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
    # Claude sometimes drops the underscores from the delimiter
    # (<<<NEWMONITORINGITEMS>>> instead of <<<NEW_MONITORING_ITEMS>>>) despite
    # the prompt spelling it out — same two variants as above, underscore-free.
    r"<<<NEWMONITORINGITEMS>>>(.*?)<<<ENDMONITORINGITEMS>>>",
    r"<<<NEWMONITORINGITEMS>>>(.*)\Z",
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


def synthesize_full_brief(
    all_sections_text: str,
    regime_flags: list[str] | None = None,
    monitoring_items: list[dict] | None = None,
    max_tokens: int = 700,
) -> str:
    """Closing synthesis for the full 5-section brief, same structured-density
    format as `synthesize_section()` (see `SYNTHESIS_MAX_TOKENS`).

    `monitoring_items` and `max_tokens` are caller-controlled — build_morning_brief()
    passes the full brief's accumulated monitoring list here since this call sees
    the whole morning's data, not just one section's. Cached `SYNTHESIS_CACHE_HOURS`
    hours by content hash.
    """
    data_hash = hashlib.md5(all_sections_text.encode()).hexdigest()
    path = _cache_path("full_brief", data_hash)
    cached = _load_cache(path)
    if cached:
        return cached

    regime_str = ", ".join(regime_flags) if regime_flags else "none"
    positions_ctx = _get_positions_context()

    monitoring_str = ""
    if monitoring_items:
        active = [i for i in monitoring_items if i.get("priority") in ("high", "medium")][:5]
        if active:
            monitoring_str = "\nACTIVE MONITORING:\n" + "\n".join(
                f'[{i["ticker"]}] {i["item"][:60]} ({i.get("age_days", 0)}d)' for i in active
            )

    prompt = f"""{FRAMEWORK}
Regime: {regime_str}

{positions_ctx}
{monitoring_str}

BRIEF DATA (last 2500 chars):
{all_sections_text[-2500:]}

Respond in EXACTLY this format:

OVERALL: [2 sentences — dominant theme and most important cross-section signal]

TOP IMPLICATIONS:
[TICKER] — [specific implication ≤12 words]
[TICKER] — [specific implication ≤12 words]
[TICKER] — [specific implication ≤12 words]

TODAY'S FOCUS:
Position: [ticker + specific action]
Monitor: [ticker/signal + specific condition]

RISK: [1 sentence — single biggest portfolio risk right now]

<<<NEW_MONITORING_ITEMS>>> (use exactly this — include underscores)
[TICKER] | [specific measurable condition] | [high/medium/low]
(1-3 items, or NONE)
<<<END_MONITORING_ITEMS>>> (use exactly this — include underscores)"""

    try:
        r = client.messages.create(model=MODEL, max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}])
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
