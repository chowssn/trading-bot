"""News triage against each position's thesis breakers.

For every ticker, pulls recent headlines via `yf.Ticker(ticker).news` and
checks each title against that ticker's `thesis_breakers`
(`equity.config.positions.get_thesis_breakers()`) — the specific,
falsifiable conditions defined per-position that would invalidate the
thesis (see `positions.py` module docstring).

Matching is deliberately simple for now: a title "matches" a thesis_breaker
phrase if 2+ of the phrase's significant words (i.e. not in `_STOPWORDS`)
also appear as whole words in the title, case-insensitive. This is a
keyword screen, not NLP — it will both miss paraphrased coverage and
occasionally false-positive on an unrelated headline that happens to share
two words with a breaker phrase. Treat `thesis_breaker_match` as "worth a
human look," not as confirmation the thesis actually broke.

`yf.Ticker.news` items nest most fields under `item['content']` as of
yfinance 1.5.x; `_extract_headline()` tolerates the older flat schema too
in case that changes again.
"""

import logging
import re
from datetime import date

import yfinance as yf

from equity.config import positions as positions_config
from equity.config.market_config import (
    NEWS_MAX_TIER3_PER_TICKER,
    NEWS_SOURCE_TIER1,
    NEWS_SOURCE_TIER2,
)

logger = logging.getLogger(__name__)

_DIVIDER = "━━━━━━━━━━━━━━━━━━━━━━━━"

MAX_HEADLINES_SHOWN = 3

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "has", "have", "for", "in",
    "of", "to", "and", "or", "with", "on", "at",
}

_WORD_RE = re.compile(r"[a-z0-9']+")


def _significant_words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS}


def _match_thesis_breaker(title: str, thesis_breakers: list[str]) -> str | None:
    """Return the first thesis_breaker phrase sharing >=2 significant words with `title`, or None."""
    title_words = _significant_words(title)
    for breaker in thesis_breakers:
        if len(_significant_words(breaker) & title_words) >= 2:
            return breaker
    return None


def _fetch_news(ticker: str) -> list[dict]:
    try:
        raw = yf.Ticker(ticker).news
    except Exception as exc:  # yfinance can raise a variety of things on network/format issues
        logger.warning("News fetch failed for %s: %s", ticker, exc)
        return []
    return raw or []


def _get_source_tier(publisher: str) -> int:
    """1 (top-tier wire/paper), 2 (industry trade press), or 3 (everything else — 'unverified')."""
    pub_lower = publisher.lower() if publisher else ""
    for source in NEWS_SOURCE_TIER1:
        if source in pub_lower:
            return 1
    for source in NEWS_SOURCE_TIER2:
        if source in pub_lower:
            return 2
    return 3


def _extract_headline(item: dict) -> dict | None:
    """Pull title/published/url/publisher out of one yf.Ticker.news item, or None if unusable."""
    content = item.get("content", item)  # tolerate the pre-1.5 flat schema too
    title = content.get("title")
    if not title:
        return None

    url = (
        (content.get("canonicalUrl") or {}).get("url")
        or (content.get("clickThroughUrl") or {}).get("url")
        or content.get("link")
        or ""
    )
    publisher = (content.get("provider") or {}).get("displayName") or content.get("publisher") or ""

    return {
        "title": title,
        "published": content.get("pubDate") or content.get("displayTime") or "",
        "url": url,
        "publisher": publisher,
        "source_tier": _get_source_tier(publisher),
    }


def _cap_tier3(headlines: list[dict]) -> list[dict]:
    """All tier-1/2 headlines, plus at most `NEWS_MAX_TIER3_PER_TICKER` tier-3 ones, in original order."""
    kept = []
    tier3_kept = 0
    for h in headlines:
        if h.get("source_tier") == 3:
            if tier3_kept >= NEWS_MAX_TIER3_PER_TICKER:
                continue
            tier3_kept += 1
        kept.append(h)
    return kept


def run_news_triage(tickers: list[str]) -> dict:
    """Fetch and thesis-breaker-check recent news for each ticker in `tickers`.

    `headlines` is capped at `NEWS_MAX_TIER3_PER_TICKER` tier-3 (lowest
    credibility) headlines per ticker — tier-1/2 sources are never capped.
    `all_headlines` is the full, untruncated list (used for the Telegram
    headline-pagination flow — see `equity.telegram.formatters`).

    A ticker with no news, or whose fetch fails, gets empty
    `headlines`/`all_headlines` lists rather than being dropped from the
    result — never crash the whole run over one bad ticker.
    """
    result = {}

    for ticker in tickers:
        thesis_breakers = positions_config.get_thesis_breakers(ticker)
        all_headlines = []
        thesis_alerts = []

        for item in _fetch_news(ticker):
            headline = _extract_headline(item)
            if headline is None:
                continue

            matched = _match_thesis_breaker(headline["title"], thesis_breakers)
            headline["thesis_breaker_match"] = matched is not None
            headline["matched_breaker"] = matched
            if matched and matched not in thesis_alerts:
                thesis_alerts.append(matched)

            all_headlines.append(headline)

        headlines = _cap_tier3(all_headlines)

        result[ticker] = {
            "headlines": headlines,
            "all_headlines": all_headlines,
            "has_thesis_alert": bool(thesis_alerts),
            "thesis_alerts": thesis_alerts,
        }

    return result


def _format_title(headline: dict) -> str:
    """Markdown link `[Title](url)` when a URL is available, else the bare title."""
    title, url = headline["title"], headline.get("url")
    return f"[{title}]({url})" if url else title


def _format_headline_line(headline: dict, *, prefix: str = "  • ") -> str:
    tier = headline.get("source_tier", 3)
    marker = " ⚠️" if headline.get("thesis_breaker_match") else ""
    unverified = " [unverified]" if tier == 3 else ""
    publisher = headline.get("publisher") or ""
    pub_str = f" [{publisher}]" if publisher and tier != 3 else ""
    return f"{prefix}{_format_title(headline)}{pub_str}{unverified}{marker}"


def format_news_triage(triage_data: dict) -> str:
    """Render `run_news_triage()`'s output dict as a Telegram-ready string."""
    lines = [f"📰 NEWS TRIAGE — {date.today().strftime('%b %d, %Y')}", _DIVIDER]

    alert_tickers = [t for t, d in triage_data.items() if d.get("has_thesis_alert")]
    if alert_tickers:
        lines.append("⚠️ THESIS ALERTS")
        for ticker in alert_tickers:
            data = triage_data[ticker]
            for breaker in data["thesis_alerts"]:
                lines.append(f"{ticker}: '{breaker}'")
                matched = [h for h in data["headlines"] if h.get("matched_breaker") == breaker]
                for h in matched:
                    # Tier 3 (unverified) source matching a thesis-breaker phrase
                    # is flagged louder — worth a skeptical look before acting.
                    prefix = "  ⚠️ LOW CREDIBILITY — '" if h.get("source_tier") == 3 else "  → '"
                    lines.append(f"{prefix}{h['title']}'")
        lines.append("")

    lines.append("📋 HEADLINES")
    for ticker, data in triage_data.items():
        headlines = data.get("headlines", [])
        if not headlines:
            lines.append(ticker)
            lines.append("  No recent news")
            continue

        lines.append(f"{ticker} ({len(headlines)} article{'s' if len(headlines) != 1 else ''})")
        for headline in headlines[:MAX_HEADLINES_SHOWN]:
            lines.append(_format_headline_line(headline))

    lines.append(_DIVIDER)
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    triage_result = run_news_triage(positions_config.get_all_tickers())
    print(format_news_triage(triage_result))
