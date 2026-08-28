"""Manual equity holdings and watchlist.

This file is hand-maintained. `POSITIONS` holds names actually owned;
`WATCHLIST` holds names being monitored but not yet (fully) entered.

`thesis_breakers` drives `news_triage` alerts — keep these specific and
falsifiable so an alert can actually fire on them, not vague sentiment.

Fields marked TODO below need real numbers filled in (entry date/price,
position size, thesis date) — they were not specified when this file was
generated and are placeholders.
"""

POSITIONS = {
    "APP": {
        # Position details
        "entry_date": "YYYY-MM",  # TODO: fill in actual entry date
        "entry_price": 0.00,  # TODO: fill in actual entry price
        "current_size_pct": 0.0,  # TODO: fill in % of portfolio
        "tier": "high_conviction",

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
    },
    "CCJ": {
        # Position details
        "entry_date": "YYYY-MM",  # TODO: fill in actual entry date
        "entry_price": 0.00,  # TODO: fill in actual entry price
        "current_size_pct": 0.0,  # TODO: fill in % of portfolio
        "tier": "core",

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
    },
    "CEG": {
        # Position details
        "entry_date": "YYYY-MM",  # TODO: fill in actual entry date
        "entry_price": 0.00,  # TODO: fill in actual entry price
        "current_size_pct": 0.0,  # TODO: fill in % of portfolio
        "tier": "core",

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
    },
    "MSFT": {
        # Position details
        "entry_date": "YYYY-MM",  # TODO: fill in actual entry date
        "entry_price": 0.00,  # TODO: fill in actual entry price
        "current_size_pct": 0.0,  # TODO: fill in % of portfolio
        "tier": "core",

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
    },
    "UMAC": {
        # Position details
        "entry_date": "YYYY-MM",  # TODO: fill in actual entry date
        "entry_price": 0.00,  # TODO: fill in actual entry price
        "current_size_pct": 0.0,  # TODO: fill in % of portfolio
        "tier": "speculative",

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
    },
    "PGR": {
        # Position details
        "entry_date": "YYYY-MM",  # TODO: fill in actual entry date
        "entry_price": 0.00,  # TODO: fill in actual entry price
        "current_size_pct": 0.0,  # TODO: fill in % of portfolio
        "tier": "core",

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
    },
}


WATCHLIST = {
    "APP": {
        # Position details
        "entry_date": "YYYY-MM",  # not yet entered
        "entry_price": 0.00,  # limit orders opened, not yet filled
        "current_size_pct": 0.0,
        "tier": "watchlist",

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
    },
}


def get_all_tickers() -> list[str]:
    return list(POSITIONS.keys()) + list(WATCHLIST.keys())


def get_position(ticker: str) -> dict | None:
    return POSITIONS.get(ticker) or WATCHLIST.get(ticker)


def get_thesis_breakers(ticker: str) -> list[str]:
    pos = get_position(ticker)
    return pos.get("thesis_breakers", []) if pos else []
