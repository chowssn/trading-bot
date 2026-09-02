"""Portfolio positions and watchlist configuration.

IMPORTANT: This file is version-controlled — every change is tracked in git.
Do not edit manually outside of approved AI-assisted discussions.
Position sizes and entry prices come from IBKR API (Module 4 — not yet connected).
Thesis entries are AI-assisted drafts approved by the portfolio manager.
See config/changelog.md for human-readable change history.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_OVERRIDE_PATH = Path(__file__).with_name("positions_override.json")


POSITIONS = {
    "CCJ": {
        "tier": "core",
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
}


WATCHLIST = {
    "APP": {
        "tier": "watchlist",
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


def _apply_overrides(base: dict, overrides: dict) -> dict:
    """Merge per-ticker field overrides on top of `base`; new tickers pass through as-is."""
    merged = dict(base)
    for ticker, fields in overrides.items():
        merged[ticker] = {**merged.get(ticker, {}), **fields}
    return merged


_override = _load_override()
POSITIONS = _apply_overrides(POSITIONS, _override["POSITIONS"])
WATCHLIST = _apply_overrides(WATCHLIST, _override["WATCHLIST"])


def get_all_tickers() -> list[str]:
    return list(POSITIONS.keys()) + list(WATCHLIST.keys())


def get_position(ticker: str) -> dict | None:
    return POSITIONS.get(ticker) or WATCHLIST.get(ticker)


def get_thesis_breakers(ticker: str) -> list[str]:
    pos = get_position(ticker)
    return pos.get("thesis_breakers", []) if pos else []
