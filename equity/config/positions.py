"""Portfolio positions and watchlist configuration.

IMPORTANT: This file is version-controlled — every change is tracked in git.
Do not edit manually outside of approved AI-assisted discussions.
Position sizes and entry prices come from IBKR API (Module 4 — not yet connected).
Thesis entries are AI-assisted drafts approved by the portfolio manager.
See config/changelog.md for human-readable change history.
"""

import copy
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_OVERRIDE_PATH = Path(__file__).with_name("positions_override.json")


POSITIONS = {
    "CCJ": {
        "tier": "core",
        "tier_v2": "core_macro",
        "style": "macro",
        "classification_status": "complete",
        "size_target_pct": 4.0,
        "thesis_source": "ai_assisted",
        "last_reviewed": "2026-08",

        # Thesis
        "thesis": (
            "Nuclear energy required for AI and datacenter power demand. "
            "Supply-constrained uranium market with long contract cycles "
            "limiting new supply response."
        ),
        "thesis_date": "2026-08",

        # What would break the thesis — these drive news_triage alerts
        "thesis_breakers": [
            "Government policy reversal on nuclear permitting",
            "AI datacenter power demand forecast cuts >20%",
            "Major new uranium supply announcement from tier-1 jurisdiction",
        ],

        # Sector and macro context
        "sector": "Energy",
        "macro_thesis": (
            "AI infrastructure buildout requires reliable baseload power. "
            "Nuclear is the only viable large-scale solution."
        ),

        # Exit notes
        "target_exit_conditions": "TODO: define price/thesis-based exit conditions",
        "stop_thesis": False,

        # Optional: tickers considered meaningful peers (style/sub-sector)
        # even before they show high price correlation — surfaced by the
        # screener's correlation gate (see market_config.CORRELATION_CATEGORIES).
        "peer_tickers": [],
    },
    "CEG": {
        "tier": "core",
        "tier_v2": "core_macro",
        "style": "macro",
        "classification_status": "complete",
        "size_target_pct": 4.0,
        "thesis_source": "ai_assisted",
        "last_reviewed": "2026-08",

        # Thesis
        "thesis": (
            "Pure-play nuclear operator benefiting from AI datacenter power "
            "demand. Microsoft and other hyperscaler contracts provide "
            "revenue visibility."
        ),
        "thesis_date": "2026-08",

        # What would break the thesis — these drive news_triage alerts
        "thesis_breakers": [
            "Hyperscaler power contract cancellations",
            "Nuclear regulatory setback on specific plants",
            "Natural gas price collapse making nuclear uncompetitive on new contracts",
        ],

        # Sector and macro context
        "sector": "Utilities",
        "macro_thesis": "Same as CCJ — nuclear renaissance driven by AI power demand.",

        # Exit notes
        "target_exit_conditions": "TODO: define price/thesis-based exit conditions",
        "stop_thesis": False,

        # Optional: tickers considered meaningful peers (style/sub-sector)
        # even before they show high price correlation — surfaced by the
        # screener's correlation gate (see market_config.CORRELATION_CATEGORIES).
        "peer_tickers": [],
    },
    "MSFT": {
        "tier": "core",
        "tier_v2": "core_compounder",
        "style": "growth",
        "classification_status": "needs_thesis",
        "size_target_pct": 5.0,
        "thesis_source": "ai_assisted",
        "last_reviewed": "2026-08",

        # Thesis
        "thesis": (
            "Cloud infrastructure and AI platform leader. Azure growth "
            "driven by enterprise AI adoption. Recent price weakness at "
            "52-week low provides entry."
        ),
        "thesis_date": "2026-08",

        # What would break the thesis — these drive news_triage alerts
        "thesis_breakers": [
            "Azure revenue growth decelerates below 20% for 2+ quarters",
            "AI Copilot adoption metrics disappoint materially",
            "Antitrust action affecting cloud or gaming business",
        ],

        # Sector and macro context
        "sector": "Information Technology",
        "macro_thesis": (
            "Enterprise AI adoption drives cloud spend. MSFT best "
            "positioned with both infrastructure (Azure) and application "
            "layer (Copilot, Office)."
        ),

        # Exit notes
        "target_exit_conditions": "TODO: define price/thesis-based exit conditions",
        "stop_thesis": False,

        # Optional: tickers considered meaningful peers (style/sub-sector)
        # even before they show high price correlation — surfaced by the
        # screener's correlation gate (see market_config.CORRELATION_CATEGORIES).
        "peer_tickers": [],
    },
    "UMAC": {
        "tier": "speculative",
        "tier_v2": "speculative_high",
        "style": "macro",
        "classification_status": "complete",
        "size_target_pct": 2.0,
        "thesis_source": "ai_assisted",
        "last_reviewed": "2026-08",

        # Thesis
        "thesis": (
            "US drone manufacturing opportunity. Drones extensively used "
            "in warfare and commercially in China but no comparable US "
            "equivalent. Regulatory tailwind from defense procurement "
            "shift away from DJI. Microcap with high risk."
        ),
        "thesis_date": "2026-08",

        # What would break the thesis — these drive news_triage alerts
        "thesis_breakers": [
            "US regulatory environment turns against domestic drone mandate",
            "DJI ban reversed or weakened",
            "Failed to secure meaningful defense contracts within 12 months",
        ],

        # Sector and macro context
        "sector": "Industrials",
        "macro_thesis": (
            "Defense spending shift toward autonomous systems. "
            "Geopolitical environment favors domestic drone manufacturing."
        ),

        # Exit notes
        "target_exit_conditions": "TODO: define price/thesis-based exit conditions",
        "stop_thesis": False,

        # Optional: tickers considered meaningful peers (style/sub-sector)
        # even before they show high price correlation — surfaced by the
        # screener's correlation gate (see market_config.CORRELATION_CATEGORIES).
        "peer_tickers": [],
    },
    "PGR": {
        "tier": "core",
        "tier_v2": "core_compounder",
        "style": "defensive",
        "classification_status": "needs_thesis",
        "size_target_pct": 5.0,
        "thesis_source": "ai_assisted",
        "last_reviewed": "2026-08",

        # Thesis
        "thesis": (
            "Best-in-class auto insurer with superior loss ratio and "
            "pricing discipline. Benefits from hard insurance market and "
            "rate increases working through."
        ),
        "thesis_date": "2026-08",

        # What would break the thesis — these drive news_triage alerts
        "thesis_breakers": [
            "Combined ratio deteriorates above 96% for 2+ quarters",
            "Cat loss event exceeding reserves",
            "Competitive pricing pressure from State Farm or Geico recovery",
        ],

        # Sector and macro context
        "sector": "Financials",
        "macro_thesis": (
            "Insurance pricing cycle favorable. Rate increases earned "
            "through. Inflation moderation improving loss ratios."
        ),

        # Exit notes
        "target_exit_conditions": "TODO: define price/thesis-based exit conditions",
        "stop_thesis": False,

        # Optional: tickers considered meaningful peers (style/sub-sector)
        # even before they show high price correlation — surfaced by the
        # screener's correlation gate (see market_config.CORRELATION_CATEGORIES).
        "peer_tickers": [],
    },
    # APP moved from watchlist — executed position. Full data (avg_cost,
    # shares, size_pct, market_value, tier) is in positions_override.json.
    "APP": {
        "tier": "high_conviction",
        "tier_v2": "core_compounder",
        "style": "growth",
        "classification_status": "complete",
        "size_target_pct": 5.0,
        "thesis_source": "ai_assisted",
        "last_reviewed": "2026-08",

        # Thesis
        "thesis": (
            "Multiple compression on growth narrative reset, not fundamental "
            "deterioration. E-commerce ramp timing issue, not structural "
            "ceiling. FCF, margins, EPS trajectory remain strong. RSI near "
            "30, deeply oversold."
        ),
        "thesis_date": "2026-08",

        # What would break the thesis — these drive news_triage alerts
        "thesis_breakers": [
            "E-commerce pixel growth fails to scale beyond gaming over 2+ quarters",
            "Gaming market share drops below 40%",
            "FCF margin deteriorates below 40%",
            "Management changes key engineering leadership",
        ],

        # Sector and macro context
        "sector": "Communication Services",
        "macro_thesis": (
            "Ad tech benefits from AI-driven targeting improvements. "
            "E-commerce TAM 3x gaming."
        ),

        # Exit notes
        "target_exit_conditions": "TODO: define price/thesis-based exit conditions",
        "stop_thesis": False,

        # Optional: tickers considered meaningful peers (style/sub-sector)
        # even before they show high price correlation — surfaced by the
        # screener's correlation gate (see market_config.CORRELATION_CATEGORIES).
        "peer_tickers": [],
    },
}


WATCHLIST = {}

# Hardcoded base state, captured before any override is applied. reload()
# always rebuilds POSITIONS/WATCHLIST starting from these copies rather
# than merging onto whatever the previous call produced, so a ticker
# removed from positions_override.json actually disappears again instead
# of lingering from an earlier merge. Deep-copied so nothing downstream
# (a merge, or code that mutates a position dict in place) can reach back
# and corrupt the base snapshot through a shared nested list/dict.
_BASE_POSITIONS = copy.deepcopy(POSITIONS)
_BASE_WATCHLIST = copy.deepcopy(WATCHLIST)

# The live dicts. Other modules hold references to these same objects
# (`from equity.config.positions import POSITIONS`, or
# `positions_config.POSITIONS`) — _apply_overrides() below mutates them
# in place via .clear()/.update() rather than rebinding the names, so
# every existing reference sees a reload() without needing to re-import.
# Deep-copied from base for the same reason as above.
POSITIONS = copy.deepcopy(_BASE_POSITIONS)
WATCHLIST = copy.deepcopy(_BASE_WATCHLIST)


def _load_override() -> dict:
    """Load positions_override.json, tolerating a missing or malformed file.

    `config_manager.save_thesis_update()` and `.add_to_watchlist()` write
    this file directly (rather than editing this module's source) so
    pending, AI-assisted thesis changes can be committed without parsing
    Python source. See `equity.config.config_manager`.
    """
    if not _OVERRIDE_PATH.exists():
        return {"POSITIONS": {}, "WATCHLIST": {}}
    try:
        with open(_OVERRIDE_PATH) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load positions_override.json — ignoring overrides: %s", exc)
        return {"POSITIONS": {}, "WATCHLIST": {}}
    return {"POSITIONS": data.get("POSITIONS", {}), "WATCHLIST": data.get("WATCHLIST", {})}


def _merge_overrides(base: dict, overrides: dict) -> dict:
    """Merge per-ticker field overrides on top of `base`; new tickers pass through as-is."""
    merged = dict(base)
    for ticker, fields in overrides.items():
        merged[ticker] = {**merged.get(ticker, {}), **fields}
    return merged


def _apply_overrides() -> None:
    """(Re)build POSITIONS and WATCHLIST from the hardcoded base dicts plus
    whatever is currently on disk in positions_override.json.

    Mutates the existing POSITIONS/WATCHLIST dict objects in place
    (.clear() + .update()) rather than rebinding the module-level names.
    Reassignment (`POSITIONS = new_dict`) would only repoint this module's
    own name at a new object — every other module that already holds a
    reference to the old dict (`from equity.config.positions import
    POSITIONS`, or `positions_config.POSITIONS`) would keep seeing the
    stale one. Mutating in place means every existing reference, anywhere
    in the codebase, sees the update — no re-import required.

    Safe to call any number of times — it always starts from
    `_BASE_POSITIONS`/`_BASE_WATCHLIST` and re-reads the override file each
    time, rather than merging onto the previous result, so this is
    idempotent and a removed override actually takes effect.
    """
    override = _load_override()
    merged_positions = _merge_overrides(_BASE_POSITIONS, override["POSITIONS"])
    merged_watchlist = _merge_overrides(_BASE_WATCHLIST, override["WATCHLIST"])

    POSITIONS.clear()
    POSITIONS.update(merged_positions)
    WATCHLIST.clear()
    WATCHLIST.update(merged_watchlist)


_apply_overrides()


def reload() -> None:
    """Re-read positions_override.json and rebuild POSITIONS/WATCHLIST.

    Call after any write to positions_override.json to see the change
    immediately, without restarting the bot. `config_manager.py`'s
    `save_thesis_update()`, `add_to_watchlist()`, and
    `remove_from_watchlist()` all call this after committing their write.

    Also invalidates `advisor.py`'s cached portfolio-context string (used
    in the Claude system prompt) so the next chat message reflects the
    change instead of serving a snapshot that's stale for up to 15 more
    minutes. Import is deferred to call time to avoid a circular import
    (advisor.py imports this module at its own module scope) and wrapped
    in try/except so a failure here never blocks the actual reload.
    """
    _apply_overrides()
    try:
        import equity.telegram.advisor as advisor_module
        advisor_module._portfolio_context_cache["timestamp"] = 0
    except Exception:
        pass


def get_all_tickers() -> list[str]:
    reload()
    return list(POSITIONS.keys()) + list(WATCHLIST.keys())


def get_position(ticker: str) -> dict | None:
    reload()
    return POSITIONS.get(ticker) or WATCHLIST.get(ticker)


def get_thesis_breakers(ticker: str) -> list[str]:
    pos = get_position(ticker)
    return pos.get("thesis_breakers", []) if pos else []
