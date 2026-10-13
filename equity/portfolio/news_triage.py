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

import json
import logging
import os
import re
import time
from datetime import date, timedelta
from pathlib import Path

import requests
import yfinance as yf
from dotenv import load_dotenv

from equity.config import positions as positions_config
from equity.config.market_config import (
    EDGAR_8K_ITEMS_OF_INTEREST,
    EDGAR_CACHE_HOURS,
    EDGAR_FORMS_OF_INTEREST,
    EDGAR_LOOKBACK_DAYS,
    EDGAR_MAX_FILINGS_PER_TICKER,
    EDGAR_MAX_TICKERS,
    EDGAR_REQUEST_DELAY_SECONDS,
    NEWS_MAX_TIER3_PER_TICKER,
    NEWS_SOURCE_TIER1,
    NEWS_SOURCE_TIER2,
)

load_dotenv()

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


# ----------------------------------------------------------------------
# SEC EDGAR filings
# ----------------------------------------------------------------------

# SEC requires a declared User-Agent on every request to both sec.gov
# and data.sec.gov. Without one the response is an HTTP 403 "Your
# Request Originates from an Undeclared Automated Tool" HTML page, not
# JSON — verified against both hosts. SEC's fair-access policy asks for
# contact info in it; a descriptive UA with no address is accepted today
# (also verified), so the default here carries no personal data and
# SEC_EDGAR_USER_AGENT in .env can add a contact address if you'd rather
# be reachable than anonymous to them.
_EDGAR_USER_AGENT = os.getenv("SEC_EDGAR_USER_AGENT", "trading-bot/1.0 (personal research)")
_EDGAR_HEADERS = {
    "User-Agent": _EDGAR_USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
}
_EDGAR_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
_EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_EDGAR_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"
_EDGAR_TIMEOUT_SECONDS = 15

_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache"
_EDGAR_TICKER_MAP_PATH = _CACHE_DIR / "edgar_ticker_map.json"
_EDGAR_FILINGS_PATH = _CACHE_DIR / "edgar_filings.json"
_EDGAR_TICKER_MAP_CACHE_HOURS = 24 * 7   # new listings are rare


def _cache_age_hours(path: Path) -> float | None:
    """Age of `path` in hours, or None if it doesn't exist."""
    if not path.exists():
        return None
    return (time.time() - path.stat().st_mtime) / 3600


def _fetch_edgar_ticker_map() -> dict[str, int]:
    """`{TICKER: cik}` from SEC's company_tickers.json, cached a week.

    EDGAR is keyed by CIK, not ticker, so this map is the required first
    hop for the submissions API. Returns {} on failure.
    """
    age = _cache_age_hours(_EDGAR_TICKER_MAP_PATH)
    if age is not None and age < _EDGAR_TICKER_MAP_CACHE_HOURS:
        try:
            return json.loads(_EDGAR_TICKER_MAP_PATH.read_text())
        except Exception:
            logger.debug("EDGAR ticker map cache unreadable — refetching", exc_info=True)

    try:
        resp = requests.get(
            _EDGAR_TICKER_MAP_URL, headers=_EDGAR_HEADERS, timeout=_EDGAR_TIMEOUT_SECONDS
        )
        resp.raise_for_status()
        raw = resp.json()
    except Exception as exc:
        logger.warning("_fetch_edgar_ticker_map failed: %s", exc)
        return {}

    # Shape is {"0": {"cik_str": 1045810, "ticker": "NVDA", ...}, ...}
    ticker_map = {
        entry["ticker"].upper(): int(entry["cik_str"])
        for entry in raw.values()
        if entry.get("ticker") and entry.get("cik_str") is not None
    }
    try:
        _EDGAR_TICKER_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
        _EDGAR_TICKER_MAP_PATH.write_text(json.dumps(ticker_map))
    except Exception:
        logger.debug("Could not write EDGAR ticker map cache", exc_info=True)
    return ticker_map


def _describe_8k_items(items: str) -> str:
    """Human-readable 8-K item codes, e.g. "2.02,9.01" -> "results of operations".

    Codes with no entry in EDGAR_8K_ITEMS_OF_INTEREST (9.01 "financial
    statements and exhibits", which rides along on most 8-Ks) are
    dropped rather than printed as bare numbers.
    """
    described = [
        EDGAR_8K_ITEMS_OF_INTEREST[code.strip()]
        for code in items.split(",")
        if code.strip() in EDGAR_8K_ITEMS_OF_INTEREST
    ]
    return "; ".join(dict.fromkeys(described))


def _fetch_ticker_filings(ticker: str, cik: int, cutoff: str) -> list[dict]:
    """Filings of interest for one CIK filed on/after `cutoff` (ISO date)."""
    try:
        resp = requests.get(
            _EDGAR_SUBMISSIONS_URL.format(cik=cik),
            headers=_EDGAR_HEADERS,
            timeout=_EDGAR_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        recent = resp.json()["filings"]["recent"]
    except Exception as exc:
        logger.warning("EDGAR submissions fetch failed for %s (CIK %s): %s", ticker, cik, exc)
        return []

    forms = recent.get("form", [])
    filings = []
    for i, form in enumerate(forms):
        if form not in EDGAR_FORMS_OF_INTEREST:
            continue

        filing_date = recent["filingDate"][i]
        # `recent` is newest-first, so the first filing older than the
        # cutoff means every remaining one is too.
        if filing_date < cutoff:
            break

        accession = recent["accessionNumber"][i]
        document = recent.get("primaryDocument", [""] * len(forms))[i]
        items = recent.get("items", [""] * len(forms))[i] or ""

        filings.append({
            "form": form,
            "form_label": EDGAR_FORMS_OF_INTEREST[form],
            "date": filing_date,
            "report_date": recent.get("reportDate", [""] * len(forms))[i] or "",
            "items": _describe_8k_items(items) if items else "",
            "accession": accession,
            # Archive paths use the accession number with dashes stripped
            # and an unpadded CIK — a padded CIK or a dashed accession
            # both 404.
            "url": _EDGAR_ARCHIVE_URL.format(
                cik=cik, accession=accession.replace("-", ""), document=document
            ) if document else (
                f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}"
            ),
        })
        if len(filings) >= EDGAR_MAX_FILINGS_PER_TICKER:
            break

    return filings


def fetch_edgar_filings(
    tickers: list[str], days_back: int = EDGAR_LOOKBACK_DAYS
) -> dict[str, list[dict]]:
    """Recent supply/demand-relevant SEC filings per ticker.

    Covers 8-K (material events), S-3/S-3ASR/S-1 (registrations),
    424B3/424B5 (prospectus takedowns off a shelf) and Form 4 (insider
    transactions) — see `EDGAR_FORMS_OF_INTEREST`. Free, no API key.

    Uses the **submissions API** (`data.sec.gov/submissions/CIK*.json`),
    which is scoped to the filer, rather than EDGAR full-text search
    (`efts.sec.gov`). Full-text search matches filing *text*, so a query
    for a ticker returns every filing that merely mentions it — a test
    query for "AAPL" returned 434 hits, including an unrelated issuer's
    exhibit. That is the wrong tool for "what did the companies I hold
    file". The submissions route needs a ticker->CIK hop
    (`_fetch_edgar_ticker_map()`), which is why the map is cached.

    Returns `{ticker: [filing, ...]}`, omitting tickers with no filings
    of interest. Tickers are capped at `EDGAR_MAX_TICKERS` and requests
    are spaced by `EDGAR_REQUEST_DELAY_SECONDS` to respect SEC's
    fair-access policy. Never raises.

    Cached to `equity/data/cache/edgar_filings.json` for
    EDGAR_CACHE_HOURS, keyed by the ticker set + lookback so a different
    request doesn't read a stale answer.
    """
    requested = [t.upper() for t in tickers][:EDGAR_MAX_TICKERS]
    if not requested:
        return {}

    cache_key = f"{','.join(sorted(requested))}|{days_back}"
    age = _cache_age_hours(_EDGAR_FILINGS_PATH)
    if age is not None and age < EDGAR_CACHE_HOURS:
        try:
            cached = json.loads(_EDGAR_FILINGS_PATH.read_text())
            if cached.get("cache_key") == cache_key:
                return cached.get("filings", {})
        except Exception:
            logger.debug("EDGAR filings cache unreadable — refetching", exc_info=True)

    ticker_map = _fetch_edgar_ticker_map()
    if not ticker_map:
        return {}

    cutoff = (date.today() - timedelta(days=days_back)).isoformat()
    results: dict[str, list[dict]] = {}

    for ticker in requested:
        cik = ticker_map.get(ticker)
        if cik is None:
            # Normal for ETFs and non-US listings — they aren't 10-K/8-K
            # filers under a ticker-mapped CIK.
            logger.debug("fetch_edgar_filings: no CIK for %s — skipping", ticker)
            continue

        filings = _fetch_ticker_filings(ticker, cik, cutoff)
        if filings:
            results[ticker] = filings
        time.sleep(EDGAR_REQUEST_DELAY_SECONDS)

    try:
        _EDGAR_FILINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        _EDGAR_FILINGS_PATH.write_text(
            json.dumps({"cache_key": cache_key, "filings": results})
        )
    except Exception:
        logger.debug("Could not write EDGAR filings cache", exc_info=True)

    return results


def format_edgar_filings(
    edgar_data: dict[str, list[dict]], days_back: int = EDGAR_LOOKBACK_DAYS
) -> str:
    """Render `fetch_edgar_filings()`'s output. '' when there's nothing to show.

    `days_back` only labels the header — pass the same value given to
    `fetch_edgar_filings()` so the stated window matches the data.
    """
    if not edgar_data:
        return ""

    lines = [f"📄 SEC FILINGS (last {days_back} days)", _DIVIDER]
    for ticker, filings in edgar_data.items():
        lines.append(ticker)
        for f in filings:
            detail = f" — {f['items']}" if f["items"] else ""
            label = f"{f['form']} ({f['form_label']})"
            lines.append(f"  • [{label}]({f['url']}) {f['date']}{detail}")
    lines.append(_DIVIDER)
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    triage_result = run_news_triage(positions_config.get_all_tickers())
    print(format_news_triage(triage_result))
