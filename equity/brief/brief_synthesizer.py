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
    "performance":     "relative strength, sector rotation, and position vs sector divergence",
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
