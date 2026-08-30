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


def _extract_headline(item: dict) -> dict | None:
    """Pull title/published/url out of one yf.Ticker.news item, or None if unusable."""
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

    return {
        "title": title,
        "published": content.get("pubDate") or content.get("displayTime") or "",
        "url": url,
    }


def run_news_triage(tickers: list[str]) -> dict:
    """Fetch and thesis-breaker-check recent news for each ticker in `tickers`.

    A ticker with no news, or whose fetch fails, gets an empty
    `headlines` list rather than being dropped from the result — never
    crash the whole run over one bad ticker.
    """
    result = {}

    for ticker in tickers:
        thesis_breakers = positions_config.get_thesis_breakers(ticker)
        headlines = []
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

            headlines.append(headline)

        result[ticker] = {
            "headlines": headlines,
            "has_thesis_alert": bool(thesis_alerts),
            "thesis_alerts": thesis_alerts,
        }

    return result


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
                matched_titles = [
                    h["title"] for h in data["headlines"] if h.get("matched_breaker") == breaker
                ]
                for title in matched_titles:
                    lines.append(f"  → '{title}'")
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
            marker = " ⚠️" if headline.get("thesis_breaker_match") else ""
            lines.append(f"  • {headline['title']}{marker}")

    lines.append(_DIVIDER)
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    triage_result = run_news_triage(positions_config.get_all_tickers())
    print(format_news_triage(triage_result))
