"""Earnings calendar for holdings/watchlist, with an options-implied move estimate.

Two sources, merged per ticker:

- FMP's `/stable/earnings-calendar` endpoint (`_fetch_fmp_earnings()`),
  filtered down to `positions.get_all_tickers()` — only our own holdings
  and watchlist names matter here, not the whole market's earnings.

  KNOWN LIMITATION (confirmed live 2026-08-30, on this account's FMP
  plan): this endpoint's rows carry `symbol`, `date`, `epsEstimated`,
  `revenueEstimated` (and the `*Actual` counterparts once reported) but no
  time-of-day field at all — no `time`, `announcementTime`, or similar key
  appears in any sampled row. `time_of_day` is therefore always
  `'Unknown'` right now; `_TIME_OF_DAY_MAP` is still here and will pick up
  a `'bmo'`/`'amc'` value the moment FMP starts returning one (or a higher
  plan tier exposes it), no code change needed beyond confirming the field
  name.

  This endpoint also 402s outright on some FMP plans (it's not on the
  same tier as the statement endpoints `roic_calculator.py` uses) — see
  `_fetch_fmp_earnings()` and `FMP_UNAVAILABLE_MESSAGE`.

- An ATM straddle from `yfinance` options (`_fetch_implied_move()`), for
  any matched ticker reporting within `market_config.EARNINGS_LOOKAHEAD_DAYS`:
  picks the options expiration on or after the report date closest to it (an
  expiration before earnings can't price the move at all), takes the
  strike closest to the current price from that chain, and sums
  call+put `lastPrice` for the straddle. `iv30` is the average of that
  same ATM call/put pair's `impliedVolatility` — a proxy for elevated
  vol, not a literal 30-day-constant-maturity IV (there's no separate
  30-day-out chain fetch here, per the spec this module was built from).

Whole-fetch result is cached to `equity/data/cache/earnings_calendar.json`
for `CACHE_HOURS` hours, keyed by `days_ahead` so a call with a different
window never reuses a mismatched cache.
"""

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import requests
import yfinance as yf

from equity.config import positions as positions_config
from equity.config import settings
from equity.config.market_config import EARNINGS_ALERT_DAYS, EARNINGS_LOOKAHEAD_DAYS

logger = logging.getLogger(__name__)

_DIVIDER = "━━━━━━━━━━━━━━━━━━━━━━━━"

FMP_BASE_URL = "https://financialmodelingprep.com/stable"
REQUEST_TIMEOUT_SECONDS = 15
FMP_UNAVAILABLE_MESSAGE = "Earnings calendar unavailable (FMP plan required)"

_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache"
_CACHE_PATH = _CACHE_DIR / "earnings_calendar.json"
CACHE_HOURS = 6

# See module docstring — no sampled FMP row has carried a time-of-day
# field yet, so every lookup here currently falls through to 'Unknown'.
_TIME_OF_DAY_MAP = {"bmo": "BMO", "amc": "AMC"}


# ---------------------------------------------------------------------------
# Cache (6 hours, keyed by days_ahead)
# ---------------------------------------------------------------------------

def _load_cache(days_ahead: int) -> dict | None:
    if not _CACHE_PATH.exists():
        return None
    try:
        with open(_CACHE_PATH) as f:
            cache = json.load(f)
        fetched_at = datetime.fromisoformat(cache["fetched_at"])
        cached_days_ahead = cache["days_ahead"]
        payload = cache["data"]
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.warning("earnings_calendar.json cache unreadable, ignoring: %s", exc)
        return None

    if cached_days_ahead != days_ahead:
        return None
    if (datetime.now() - fetched_at).total_seconds() > CACHE_HOURS * 3600:
        return None
    return payload


def _write_cache(days_ahead: int, data: dict) -> None:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(_CACHE_PATH, "w") as f:
            json.dump({"fetched_at": datetime.now().isoformat(), "days_ahead": days_ahead, "data": data}, f, indent=2)
    except OSError as exc:
        logger.warning("Failed to write earnings_calendar.json cache: %s", exc)


# ---------------------------------------------------------------------------
# Source 1 — FMP earnings calendar
# ---------------------------------------------------------------------------

def _safe_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fetch_fmp_earnings(start: date, end: date) -> tuple[list[dict] | None, list[str]]:
    """Raw FMP earnings-calendar rows for [start, end], or None if FMP is unavailable.

    None (not []) specifically means "could not use FMP at all" — missing
    API key, a 402 (this endpoint requires a plan tier this account may
    not have), or any other request failure. Callers should show
    `FMP_UNAVAILABLE_MESSAGE` in that case rather than "no earnings
    found" — an empty list is the legitimate "FMP responded, nothing
    scheduled for our tickers" case.
    """
    warnings: list[str] = []
    if not settings.FMP_API_KEY:
        warnings.append("FMP_API_KEY not set — earnings calendar unavailable")
        return None, warnings

    url = f"{FMP_BASE_URL}/earnings-calendar"
    params = {"from": start.isoformat(), "to": end.isoformat(), "apikey": settings.FMP_API_KEY}
    try:
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        if resp.status_code == 402:
            warnings.append("FMP earnings-calendar returned 402 — plan upgrade required")
            return None, warnings
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        warnings.append(f"FMP earnings-calendar request failed: {exc}")
        return None, warnings

    if not isinstance(data, list):
        warnings.append(f"FMP earnings-calendar returned unexpected payload: {data!r}")
        return None, warnings
    return data, warnings


# ---------------------------------------------------------------------------
# Source 2 — implied earnings move from yfinance options
# ---------------------------------------------------------------------------

def _fetch_implied_move(ticker: str, report_date: date) -> tuple[float | None, float | None, str | None]:
    """(implied_move_pct, iv30, warning) — ATM straddle move and IV from the nearest options chain.

    Picks the expiration on or after `report_date` closest to it (an
    expiration before earnings can't price the move); falls back to the
    single closest expiration overall if every one available is already
    before `report_date`. Returns (None, None, warning) rather than
    raising if the ticker has no options chain or any fetch step fails.
    """
    try:
        t = yf.Ticker(ticker)
        expirations = t.options
    except Exception as exc:  # yfinance can raise a variety of things on network/format issues
        return None, None, f"{ticker}: options expirations fetch failed: {exc}"

    if not expirations:
        return None, None, f"{ticker}: no options chain available for implied move"

    exp_dates = []
    for exp_str in expirations:
        try:
            exp_dates.append(date.fromisoformat(exp_str))
        except ValueError:
            continue
    if not exp_dates:
        return None, None, f"{ticker}: could not parse any options expiration dates"

    on_or_after = [d for d in exp_dates if d >= report_date]
    target_date = min(on_or_after) if on_or_after else min(exp_dates, key=lambda d: abs((d - report_date).days))

    try:
        chain = t.option_chain(target_date.isoformat())
        current_price = t.fast_info.get("lastPrice")
    except Exception as exc:
        return None, None, f"{ticker}: option chain fetch failed for {target_date.isoformat()}: {exc}"

    calls, puts = chain.calls, chain.puts
    if calls.empty or puts.empty or not current_price:
        return None, None, f"{ticker}: options chain for {target_date.isoformat()} missing calls/puts/price"

    atm_call = calls.loc[(calls["strike"] - current_price).abs().idxmin()]
    atm_put = puts.loc[(puts["strike"] - current_price).abs().idxmin()]

    call_price = _safe_float(atm_call.get("lastPrice"))
    put_price = _safe_float(atm_put.get("lastPrice"))
    if call_price is None or put_price is None:
        return None, None, f"{ticker}: ATM straddle prices unavailable for {target_date.isoformat()}"

    implied_move_pct = (call_price + put_price) / current_price * 100

    ivs = [v for v in (_safe_float(atm_call.get("impliedVolatility")), _safe_float(atm_put.get("impliedVolatility"))) if v is not None]
    iv30 = (sum(ivs) / len(ivs)) * 100 if ivs else None

    return implied_move_pct, iv30, None


# ---------------------------------------------------------------------------
# Importance scoring
# ---------------------------------------------------------------------------

def score_earnings_importance(ticker: str, days_until: int, implied_move_pct: float | None) -> str:
    """HIGH: an owned position, reporting within `market_config.EARNINGS_ALERT_DAYS`, implied move > 5%.

    MEDIUM: on the watchlist, OR reporting in the
    (EARNINGS_ALERT_DAYS, EARNINGS_LOOKAHEAD_DAYS] window (3-7 days out by
    default), OR an implied move of 3-5%. LOW: everything else.
    `in_positions`/`in_watchlist` are looked up here (not passed in) to
    match the 3-argument signature this was specified with.
    """
    move = implied_move_pct if implied_move_pct is not None else 0.0
    in_positions = ticker in positions_config.POSITIONS
    in_watchlist = ticker in positions_config.WATCHLIST

    if in_positions and days_until <= EARNINGS_ALERT_DAYS and move > 5.0:
        return "HIGH"
    if in_watchlist or EARNINGS_ALERT_DAYS < days_until <= EARNINGS_LOOKAHEAD_DAYS or 3.0 <= move <= 5.0:
        return "MEDIUM"
    return "LOW"


def _format_alert(event: dict, report_date: date) -> str:
    days_until = event["days_until"]
    if days_until == 0:
        when = "today"
    elif days_until == 1:
        when = "tomorrow"
    else:
        when = f"in {days_until} days"
    move = f" — implied ±{event['implied_move_pct']:.1f}% move" if event["implied_move_pct"] is not None else ""
    return f"⚠️ {event['ticker']} reports {when} ({report_date.strftime('%a')} {event['time_of_day']}){move}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_earnings_calendar(days_ahead: int = EARNINGS_LOOKAHEAD_DAYS) -> dict:
    """Upcoming earnings for `positions.get_all_tickers()` within `days_ahead` days.

    Never raises: an FMP failure (including a 402) sets `fmp_available`
    False with `upcoming`/`alerts` empty rather than propagating; a failed
    implied-move fetch for one ticker just leaves that ticker's
    `implied_move_pct`/`iv30` as None and logs a warning, never drops the
    ticker from `upcoming`.
    """
    cached = _load_cache(days_ahead)
    if cached is not None:
        return cached

    today = date.today()
    end = today + timedelta(days=days_ahead)
    our_tickers = set(positions_config.get_all_tickers())

    raw_events, warnings = _fetch_fmp_earnings(today, end)
    if raw_events is None:
        result = {
            "upcoming": [],
            "alerts": [],
            "as_of": datetime.now().isoformat(),
            "days_ahead": days_ahead,
            "fmp_available": False,
            "data_warnings": warnings,
        }
        _write_cache(days_ahead, result)
        return result

    upcoming: list[dict] = []
    alerts: list[str] = []
    for row in raw_events:
        ticker = row.get("symbol")
        if ticker not in our_tickers:
            continue

        report_date_str = row.get("date")
        if not report_date_str:
            continue
        try:
            report_date = date.fromisoformat(report_date_str)
        except ValueError:
            continue

        days_until = (report_date - today).days
        if days_until < 0 or days_until > days_ahead:
            continue

        time_of_day = _TIME_OF_DAY_MAP.get(str(row.get("time") or "").lower(), "Unknown")
        eps_estimate = _safe_float(row.get("epsEstimated"))
        revenue_estimate_raw = _safe_float(row.get("revenueEstimated"))
        revenue_estimate_m = revenue_estimate_raw / 1_000_000 if revenue_estimate_raw is not None else None

        implied_move_pct = iv30 = None
        if days_until <= EARNINGS_LOOKAHEAD_DAYS:
            implied_move_pct, iv30, move_warning = _fetch_implied_move(ticker, report_date)
            if move_warning:
                warnings.append(move_warning)

        importance = score_earnings_importance(ticker, days_until, implied_move_pct)

        event = {
            "ticker": ticker,
            "report_date": report_date_str,
            "days_until": days_until,
            "time_of_day": time_of_day,
            "eps_estimate": eps_estimate,
            "revenue_estimate_m": revenue_estimate_m,
            "implied_move_pct": implied_move_pct,
            "iv30": iv30,
            "importance": importance,
            "in_positions": ticker in positions_config.POSITIONS,
            "in_watchlist": ticker in positions_config.WATCHLIST,
        }
        upcoming.append(event)

        if importance == "HIGH":
            alerts.append(_format_alert(event, report_date))

    upcoming.sort(key=lambda e: e["days_until"])

    result = {
        "upcoming": upcoming,
        "alerts": alerts,
        "as_of": datetime.now().isoformat(),
        "days_ahead": days_ahead,
        "fmp_available": True,
        "data_warnings": warnings,
    }
    _write_cache(days_ahead, result)
    return result


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _fmt_eps(value: float | None) -> str | None:
    if value is None:
        return None
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):.2f}"


def _format_report_date(report_date_str: str) -> str:
    try:
        return date.fromisoformat(report_date_str).strftime("%a %b %d")
    except ValueError:
        return report_date_str


def _format_event_row(event: dict, *, include_revenue: bool, move_label: str) -> str:
    clauses = [f"{event['ticker']}  {_format_report_date(event['report_date'])} {event['time_of_day']}"]

    eps = _fmt_eps(event["eps_estimate"])
    if eps is not None:
        clauses.append(f"EPS est: {eps}")

    if include_revenue and event["revenue_estimate_m"] is not None:
        clauses.append(f"Rev est: ${event['revenue_estimate_m']:.0f}M")

    if event["implied_move_pct"] is not None:
        clauses.append(f"±{event['implied_move_pct']:.1f}% {move_label}")

    return "  ".join(clauses)


def format_earnings_section(earnings_data: dict) -> str:
    """Render `fetch_earnings_calendar()`'s output dict as a Telegram-ready string."""
    days_ahead = earnings_data.get("days_ahead", 7)
    lines = [f"📅 EARNINGS CALENDAR (next {days_ahead} days)", _DIVIDER]

    if not earnings_data.get("fmp_available", True):
        lines.append(FMP_UNAVAILABLE_MESSAGE)
        lines.append(_DIVIDER)
        return "\n".join(lines)

    upcoming = earnings_data.get("upcoming", [])
    if not upcoming:
        lines.append(f"No earnings for your holdings in the next {days_ahead} days.")
        lines.append(_DIVIDER)
        return "\n".join(lines)

    high = [e for e in upcoming if e["importance"] == "HIGH"]
    rest = [e for e in upcoming if e["importance"] != "HIGH"]

    if high:
        lines.append("⚠️ HIGH PRIORITY")
        lines.extend(_format_event_row(e, include_revenue=True, move_label="implied move") for e in high)
        lines.append("")

    if rest:
        lines.append("📋 UPCOMING")
        lines.extend(_format_event_row(e, include_revenue=False, move_label="implied") for e in rest)
        lines.append("")

    reporting_tickers = {e["ticker"] for e in upcoming}
    if len(reporting_tickers) < len(positions_config.get_all_tickers()):
        lines.append("No other holdings reporting this week.")

    warnings = earnings_data.get("data_warnings") or []
    if warnings:
        lines.append("")
        lines.extend(f"⚠️ {w}" for w in warnings)

    lines.append(_DIVIDER)
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    earnings_result = fetch_earnings_calendar()
    print(format_earnings_section(earnings_result))

    if earnings_result.get("alerts"):
        print("\nAlerts (for a separate proactive-notification path, not shown above):")
        for alert in earnings_result["alerts"]:
            print(alert)
