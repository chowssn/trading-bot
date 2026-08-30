"""Economic calendar for the daily market brief.

Two independent sources, merged by `fetch_eco_calendar()` into one
today/tomorrow view:

- The FRED release-date calendar, filtered to
  `market_config.IMPORTANT_RELEASES` (CPI, NFP, GDP, ...). Queried via
  FRED's `/fred/releases/dates` REST endpoint directly with `requests`
  (not `fredapi` — the pinned `fredapi==0.5.2` has no `get_releases()` or
  equivalent method; this hits the same API that method would wrap, using
  the same `FRED_API_KEY`). `include_release_dates_with_no_data=true` is
  required here: a release date in the future has no data attached to it
  yet, so the default (`false`) silently returns zero upcoming releases —
  confirmed live against FRED on 2026-08-30, where a query without that
  flag returned nothing for the week ahead while the same query with it
  correctly surfaced JOLTS (2026-09-01) and the jobs report (2026-09-04).

  FRED's own release names don't all match the short display names used
  as `IMPORTANT_RELEASES` keys — `_FRED_RELEASE_NAME_MAP` bridges the four
  that differ (confirmed against FRED's `/fred/releases` list). Two
  `IMPORTANT_RELEASES` entries, ISM Manufacturing PMI and ISM Services
  PMI, have no FRED equivalent at all: ISM is proprietary data FRED does
  not carry on its release calendar. Those two will never surface through
  this source — see `_FRED_RELEASE_NAME_MAP`'s docstring.

- The FOMC meeting calendar (`fetch_fomc_dates()`), scraped from the Fed's
  own calendar page (structure confirmed live 2026-08-30 — meetings are
  `<div class="row fomc-meeting">` blocks with `.fomc-meeting__month` and
  `.fomc-meeting__date` children, grouped under `<h4><a>YYYY FOMC
  Meetings</a></h4>` year headers) and cached, with a hardcoded fallback
  for when that fetch fails.

Every threshold and lookup table here is imported from
`equity.config.market_config` — nothing FOMC- or release-related is
hardcoded in this module.
"""

import json
import logging
import os
import re
from datetime import date, datetime, timedelta
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from equity.config.market_config import (
    FOMC_CACHE_DAYS,
    FOMC_DATES_FALLBACK,
    FOMC_DATES_VALID_THROUGH,
    FOMC_FETCH_URL,
    FOMC_PROXIMITY_DAYS,
    IMPORTANT_RELEASES,
)

load_dotenv()

logger = logging.getLogger(__name__)

FRED_API_KEY = os.getenv("FRED_API_KEY", "")
FRED_RELEASES_URL = "https://api.stlouisfed.org/fred/releases/dates"
REQUEST_TIMEOUT_SECONDS = 15

_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache"
_FOMC_CACHE_PATH = _CACHE_DIR / "fomc_dates.json"

_FOMC_STALENESS_WARNING_DAYS = 60  # module docstring / market_config comment: warn 60 days before expiry

# FRED's release_name doesn't always match the plain-English key used in
# IMPORTANT_RELEASES. Confirmed against FRED's /fred/releases list
# 2026-08-30. ISM Manufacturing PMI / ISM Services PMI have no entry here
# on purpose — FRED does not carry ISM's release calendar at all (no
# amount of remapping will find it); those two IMPORTANT_RELEASES keys
# can only ever come from a different data source.
_FRED_RELEASE_NAME_MAP = {
    "Retail Sales": "Advance Monthly Sales for Retail and Food Services",
    "Industrial Production and Capacity Utilization": "G.17 Industrial Production and Capacity Utilization",
    "Housing Starts": "New Residential Construction",
    "Consumer Sentiment": "Surveys of Consumers",
}

_MONTHS = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12,
}
_FOMC_YEAR_HEADER_RE = re.compile(r"^(\d{4})\s+FOMC Meetings$")
_FOMC_DAY_RE = re.compile(r"(\d{1,2})(?:-(\d{1,2}))?")


# ---------------------------------------------------------------------------
# FRED release calendar
# ---------------------------------------------------------------------------

def _fred_release_name(display_name: str) -> str:
    return _FRED_RELEASE_NAME_MAP.get(display_name, display_name)


def _fetch_fred_release_dates(start: date, end: date) -> list[dict]:
    """Releases in `IMPORTANT_RELEASES` scheduled in [start, end] (inclusive).

    Returns a list of {'date', 'event', 'importance', 'source'}. Returns []
    on any request failure or if FRED_API_KEY isn't set — never raises.
    """
    if not FRED_API_KEY:
        logger.warning("FRED_API_KEY not set — skipping FRED release calendar")
        return []

    # Map FRED's release_name back to our IMPORTANT_RELEASES display key.
    fred_name_to_display = {_fred_release_name(name): name for name in IMPORTANT_RELEASES}

    try:
        resp = requests.get(
            FRED_RELEASES_URL,
            params={
                "api_key": FRED_API_KEY,
                "file_type": "json",
                "realtime_start": start.isoformat(),
                "realtime_end": end.isoformat(),
                # Required: a future release date has no data yet, so the
                # default (false) excludes every upcoming release — see
                # module docstring.
                "include_release_dates_with_no_data": "true",
                "sort_order": "asc",
                "limit": 1000,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        payload = resp.json()
    except (requests.RequestException, ValueError) as exc:
        logger.warning("FRED release calendar fetch failed: %s", exc)
        return []

    releases = []
    for row in payload.get("release_dates", []):
        display_name = fred_name_to_display.get(row.get("release_name"))
        if display_name is None:
            continue  # not one of IMPORTANT_RELEASES — skip
        event, importance = IMPORTANT_RELEASES[display_name]
        releases.append({"date": row["date"], "event": event, "importance": importance, "source": "FRED"})
    return releases


# ---------------------------------------------------------------------------
# FOMC calendar
# ---------------------------------------------------------------------------

def _load_fomc_cache() -> list[str] | None:
    """Return cached FOMC dates if the cache exists and is younger than FOMC_CACHE_DAYS."""
    if not _FOMC_CACHE_PATH.exists():
        return None
    try:
        with open(_FOMC_CACHE_PATH) as f:
            cache = json.load(f)
        fetched_at = datetime.fromisoformat(cache["fetched_at"])
        dates = cache["dates"]
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.warning("fomc_dates.json cache unreadable, ignoring: %s", exc)
        return None

    if (datetime.now() - fetched_at).days > FOMC_CACHE_DAYS:
        return None
    return dates


def _write_fomc_cache(dates: list[str]) -> None:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(_FOMC_CACHE_PATH, "w") as f:
            json.dump({"fetched_at": datetime.now().isoformat(), "dates": dates}, f, indent=2)
    except OSError as exc:
        logger.warning("Failed to write fomc_dates.json cache: %s", exc)


def _parse_fomc_html(html: str) -> list[str]:
    """Extract meeting end-dates (YYYY-MM-DD) from federalreserve.gov's FOMC calendar page.

    Each meeting is a `<div class="row fomc-meeting">` with `.fomc-meeting__month`
    (e.g. "January", or "Jan/Feb" for a month-spanning meeting) and
    `.fomc-meeting__date` (e.g. "27-28", "17-18*" for a projections meeting,
    or occasionally a single day like "22 (notation vote)"), grouped under
    `<h4><a>YYYY FOMC Meetings</a></h4>` year headers earlier in the page.
    The announcement date used is the last day of the meeting.
    """
    soup = BeautifulSoup(html, "html.parser")
    dates: list[str] = []
    current_year: int | None = None

    nodes = soup.find_all(
        lambda tag: (
            tag.name == "a" and _FOMC_YEAR_HEADER_RE.match(tag.get_text(strip=True) or "")
        ) or (
            tag.name == "div"
            and tag.get("class")
            and "fomc-meeting" in tag.get("class")
            and "row" in tag.get("class")
        )
    )

    for node in nodes:
        if node.name == "a":
            current_year = int(_FOMC_YEAR_HEADER_RE.match(node.get_text(strip=True)).group(1))
            continue

        if current_year is None:
            continue

        month_div = node.find(class_="fomc-meeting__month")
        date_div = node.find(class_="fomc-meeting__date")
        if month_div is None or date_div is None:
            continue

        day_match = _FOMC_DAY_RE.search(date_div.get_text(strip=True))
        if day_match is None:
            continue
        end_day = int(day_match.group(2) or day_match.group(1))

        months = month_div.get_text(strip=True).split("/")
        end_month = _MONTHS.get(months[-1].strip())
        if end_month is None:
            continue

        # A month-spanning meeting (e.g. "Jan/Feb", "31-1") only rolls into
        # the next calendar year when it spans Dec -> Jan.
        end_year = current_year
        if len(months) > 1 and end_month == 1 and _MONTHS.get(months[0].strip()) == 12:
            end_year = current_year + 1

        try:
            dates.append(date(end_year, end_month, end_day).isoformat())
        except ValueError:
            continue

    return sorted(set(dates))


def fetch_fomc_dates() -> list[str]:
    """FOMC meeting announcement dates (YYYY-MM-DD), the last day of each meeting.

    Cache-first: reuses `equity/data/cache/fomc_dates.json` if younger than
    `market_config.FOMC_CACHE_DAYS`. On a cache miss, scrapes
    `market_config.FOMC_FETCH_URL`; on any fetch or parse failure, logs a
    warning and falls back to `market_config.FOMC_DATES_FALLBACK`.
    """
    cached = _load_fomc_cache()
    if cached is not None:
        return cached

    try:
        resp = requests.get(
            FOMC_FETCH_URL,
            headers={"User-Agent": "Mozilla/5.0 (compatible; trading-bot/1.0)"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        dates = _parse_fomc_html(resp.text)
        if not dates:
            raise ValueError("no FOMC dates parsed from fetched page")
    except (requests.RequestException, ValueError) as exc:
        logger.warning("FOMC calendar fetch/parse failed, using FOMC_DATES_FALLBACK: %s", exc)
        return sorted(FOMC_DATES_FALLBACK)

    _write_fomc_cache(dates)
    return dates


def _fomc_proximity(fomc_dates: list[str], today: date) -> tuple[bool, int | None, str | None]:
    """(is_within_FOMC_PROXIMITY_DAYS, days_until_next_meeting, note) for the next upcoming FOMC date."""
    future = sorted(d for d in fomc_dates if date.fromisoformat(d) >= today)
    if not future:
        return False, None, None

    days_away = (date.fromisoformat(future[0]) - today).days
    if days_away > FOMC_PROXIMITY_DAYS:
        return False, days_away, None

    if days_away == 0:
        when = "today"
    elif days_away == 1:
        when = "tomorrow"
    else:
        when = f"in {days_away} days"
    return True, days_away, f"FOMC {when} — size reduction applies"


def _fomc_staleness_warning(fomc_dates: list[str], today: date) -> str | None:
    """Warn if FOMC_DATES_VALID_THROUGH is within 60 days and no dates beyond it were found."""
    valid_through = date.fromisoformat(FOMC_DATES_VALID_THROUGH)
    if (valid_through - today).days > _FOMC_STALENESS_WARNING_DAYS:
        return None
    if any(date.fromisoformat(d) > valid_through for d in fomc_dates):
        return None
    return (
        f"FOMC calendar may be stale — known dates only cover through "
        f"{FOMC_DATES_VALID_THROUGH} (market_config.FOMC_DATES_VALID_THROUGH) "
        f"and no dates beyond that were found"
    )


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def _drop_date(release: dict) -> dict:
    return {"event": release["event"], "importance": release["importance"], "source": release["source"]}


def fetch_eco_calendar(days_ahead: int = 1) -> dict:
    """Economic events scheduled for today and tomorrow, plus FOMC proximity.

    Never raises: a FRED or FOMC fetch failure is recorded in the returned
    `warnings` list (and logged) rather than propagated.
    """
    today = date.today()
    tomorrow = today + timedelta(days=1)
    window_end = today + timedelta(days=max(days_ahead, 1))

    warnings: list[str] = []
    if not FRED_API_KEY:
        warnings.append("FRED_API_KEY not set — economic release calendar unavailable")

    releases = _fetch_fred_release_dates(today, window_end)
    today_releases = [_drop_date(r) for r in releases if r["date"] == today.isoformat()]
    tomorrow_releases = [_drop_date(r) for r in releases if r["date"] == tomorrow.isoformat()]

    fomc_dates = fetch_fomc_dates()
    fomc_proximity, fomc_days_away, fomc_note = _fomc_proximity(fomc_dates, today)

    staleness_warning = _fomc_staleness_warning(fomc_dates, today)
    if staleness_warning:
        warnings.append(staleness_warning)

    return {
        "today": today_releases,
        "tomorrow": tomorrow_releases,
        "fomc_proximity": fomc_proximity,
        "fomc_days_away": fomc_days_away,
        "fomc_note": fomc_note,
        "warnings": warnings,
    }


def format_eco_calendar(cal: dict) -> str:
    """Render `fetch_eco_calendar()`'s output dict as a Telegram-ready string."""

    def _format_releases(releases: list[dict]) -> str:
        if not releases:
            return "No major releases"
        return " | ".join(f"{r['event']} ({r['importance']})" for r in releases)

    lines = [
        "📅 ECO CALENDAR",
        f"Today: {_format_releases(cal.get('today', []))}",
        f"Tomorrow: {_format_releases(cal.get('tomorrow', []))}",
    ]

    if cal.get("fomc_proximity") and cal.get("fomc_note"):
        lines.append(f"⚠️ {cal['fomc_note']}")

    warnings = cal.get("warnings") or []
    if warnings:
        lines.append("")
        lines.extend(f"⚠️ {w}" for w in warnings)

    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    fomc_dates_result = fetch_fomc_dates()
    print("FOMC dates:", fomc_dates_result)
    print()

    calendar_result = fetch_eco_calendar()
    print(json.dumps(calendar_result, indent=2))
    print()

    print(format_eco_calendar(calendar_result))
