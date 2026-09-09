"""Unified price cache for the equity system.

Single source of truth for current prices across the system — every
module that previously fetched prices independently (`monitor.py`,
`advisor.py`'s macro snapshot, `bot.py`'s intraday macro alerts, and now
`market_snapshot.py`'s global-signals section) reads from `price_cache`
instead. `market_snapshot.py`'s own treasury-curve/FX/commodities fetch
and `performance_tracker.py`'s multi-year return fetch are deliberately
NOT replaced by this cache (see "What this cache doesn't replace" below)
— they need history this cache doesn't keep.

One batched `yf.download()` covers everything yfinance can give in a
single call. Refreshed every 15 minutes during US market hours, every 60
minutes outside them (`PriceCache._get_ttl()`).

Coverage:
- All held positions + watchlist tickers (`positions.POSITIONS`/`WATCHLIST`)
- Sector ETFs (`SECTOR_ETFS`)
- Benchmark ETFs (`BENCHMARK_ETFS`)
- Factor ETFs (`FACTOR_ETFS`)
- FX pairs (`market_config.FX_TICKERS`)
- Commodity futures (`market_config.COMMODITY_TICKERS_EXTENDED`)
- Treasury yield proxies, yfinance side (`market_config.TREASURY_TICKERS`:
  ^IRX/^FVX/^TNX/^TYX) — see `TENOR_TO_CACHE_KEY` for the FRED side (2Y/20Y)
- Equity futures, crypto, volatility indices, international indices, and
  credit ETF proxies (`market_config.EQUITY_FUTURES` etc. — "global
  signals", added alongside the morning brief's Global Signals section)

Treasury yields (`TENOR_TO_CACHE_KEY`): 3M/5Y/10Y/30Y come from yfinance
in the same batch as everything else, scaled and tagged `is_yield=True`.
2Y/20Y aren't on yfinance at all (`market_config.TREASURY_FRED_SERIES`) —
they're fetched from FRED in the same `refresh()` cycle and stored under
their FRED series id ('DGS2'/'DGS20'), same `is_yield=True` shape. Don't
look these up by tenor name directly — use `get_yield(tenor)`, which maps
'2Y'/'5Y'/etc. to whichever cache key actually holds it.

What this cache doesn't replace:
- `market_snapshot.py`'s treasury-curve/FX/commodities fetch — those need
  ~5 years of daily history for `compute_ma_flags()` (SMA200 proximity,
  5Y high/low). This cache only ever keeps the latest 2 closes.
- `performance_tracker.py`'s benchmark/position fetch — needs the same
  multi-year history for 3Y/5Y CAGR (`_cagr()`), which a 2-close cache
  can't provide either.
- `sector_monitor.py`'s 60-day correlation matrix — needs 60 days of
  daily returns, not 2.
Wiring any of those onto this cache would silently break MA flags,
CAGR, and correlation — they keep their own longer-history fetches.
"""

import logging
import os
import threading
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import pytz

from equity.config.market_config import (
    COMMODITY_TICKERS_EXTENDED,
    CREDIT_TICKERS,
    CRYPTO_TICKERS,
    EQUITY_FUTURES,
    FX_TICKERS,
    INTERNATIONAL_INDICES,
    TREASURY_FRED_SERIES,
    TREASURY_TICKERS,
    VOLATILITY_TICKERS,
)
from equity.data.yfinance_utils import yf_download

logger = logging.getLogger(__name__)

# All sector ETFs tracked in the system
SECTOR_ETFS = [
    "XLK", "XLB", "XLF", "XLE", "XLI", "XLY", "XLC",
    "XLRE", "XLV", "XLU", "XLP", "BIL", "TLT",
]

# Benchmark ETFs
BENCHMARK_ETFS = ["SPY", "QQQ", "IWM", "EEM", "EFA"]

# Factor ETFs
FACTOR_ETFS = ["GLD", "SLV", "CPER", "SHY", "IEF", "URA", "VXX"]

# Treasury yield proxies available on yfinance (3M/5Y/10Y/30Y — see module
# docstring for 2Y/20Y, which aren't).
TREASURY_PROXIES = list(TREASURY_TICKERS.keys())  # ^IRX, ^FVX, ^TNX, ^TYX

# FX tickers
FX_LIST = list(FX_TICKERS.keys())

# Commodity futures
COMMODITY_LIST = list(COMMODITY_TICKERS_EXTENDED.keys())

# yfinance ^ticker -> yield scale. ^TNX/^TYX used to quote in tenths of a
# percent (a well-known historical yfinance quirk, and the reason
# market_snapshot.py's own scale-handling code exists) — confirmed
# empirically 2026-09-06 that this is no longer true: all four Treasury
# tickers (^IRX/^FVX/^TNX/^TYX) currently return the yield directly
# (e.g. 4.78 for a 4.78% 10Y), so no scaling is applied. If Yahoo reverts
# this, ^TNX/^TYX readings would silently come back 10x too high — the
# get_yield()/get() consumers don't sanity-range-check, so watch for a
# 10Y print like "47.8%" as the tell.
_YIELD_SCALE = {"^IRX": 1.0, "^FVX": 1.0, "^TNX": 1.0, "^TYX": 1.0}

# tenor label ('2Y', '10Y', ...) -> the cache key actually holding it:
# the yfinance ticker for TREASURY_TICKERS tenors, the FRED series id for
# TREASURY_FRED_SERIES tenors (2Y/20Y — not on yfinance). Use
# PriceCache.get_yield(tenor) rather than indexing this directly.
TENOR_TO_CACHE_KEY: dict[str, str] = {
    **{tenor: ticker for ticker, tenor in TREASURY_TICKERS.items()},
    **{tenor: series_id for series_id, tenor in TREASURY_FRED_SERIES.items()},
}

FRED_API_KEY = os.getenv("FRED_API_KEY", "")

# Instrument-type staleness thresholds for _get_session_context() — derived
# from the same config dicts _build_ticker_list() draws on rather than a
# separate hardcoded ticker list, so they can't drift out of sync with it.
_CRYPTO_TICKER_SET = set(CRYPTO_TICKERS)
_FUTURES_TICKER_SET = set(EQUITY_FUTURES) | set(COMMODITY_TICKERS_EXTENDED)
_INTL_TICKER_SET = set(INTERNATIONAL_INDICES)

# Crypto and futures (equity-index + commodity) trade well outside NYSE cash
# hours — crypto 24/7, most futures ~23h/day with a daily maintenance
# break — so gating their intraday fetch behind `is_market_hours` (the NYSE
# 9:30-16:00 ET check `refresh()` uses for everything else) meant they never
# got an intraday price outside that window and silently fell back to a
# stale settlement/daily bar overnight or the morning after a holiday. See
# `refresh()`.
_EXTENDED_INTRADAY_SET = _FUTURES_TICKER_SET | _CRYPTO_TICKER_SET

# Tickers with NO intraday history on yfinance at any interval worth
# fetching. These skip the 5m/1h fetches entirely and take their price from
# the daily bar the `daily` fetch (period="5d") already pulls for every
# ticker — so skipping costs them nothing, while attempting the 5m fetch
# costs a lot: yf_download retries an empty result 4 times with 10s/20s
# backoff, so a single dead ticker burned ~30s and four "possibly delisted"
# error lines on every 15-minute refresh cycle.
#
# Verify before adding to this set — an entry here permanently downgrades
# that ticker from a live intraday price to a daily close:
#     yf.download(TICKER, period="1d", interval="5m")
# ^VVIX and BKLN were considered and deliberately NOT added: both return
# healthy 5m data (55 and 58 bars), so they belong on the normal path.
DAILY_ONLY_TICKERS = {
    "^MOVE",   # ICE BofA MOVE Index — daily bars only, 5m returns nothing
}

_WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _extract_close(df: pd.DataFrame | None, ticker: str) -> pd.Series:
    """Close-price series for one ticker out of a (possibly multi-ticker)
    yfinance download. Empty series if `df` is None, lacks the ticker, or
    has no non-null Close values for it — callers treat "no data" and
    "ticker not present" identically either way.
    """
    if df is None:
        return pd.Series(dtype=float)
    if isinstance(df.columns, pd.MultiIndex):
        if ("Close", ticker) not in df.columns:
            return pd.Series(dtype=float)
        return df[("Close", ticker)].dropna()
    return df["Close"].dropna()


def _get_prior_close(daily_close: pd.Series, curr_price: float) -> float:
    """Prior close to compare `curr_price` against for change_1d_pct.

    Normally `iloc[-2]` — the last *complete* daily bar one back from the
    newest (see the comment at its call site for why that's always the
    right anchor). Defensive fallback: if that candidate is suspicious —
    equal to `curr_price` itself, which can happen when a stale/duplicate
    settlement bar (e.g. a thin-holiday commodity bar that just carries
    the prior session's value forward) sits at `iloc[-2]` — step back one
    more bar rather than silently reporting a 0% change that isn't real.

    Known limitation: a genuinely flat price (curr_price coincidentally
    equal to the real prior close) gets misread as "stale" by this same
    check and its prior_close silently shifted back a bar too, so change%
    ends up computed against an even older close instead of a true 0%.
    Accepted tradeoff — a wrong nonzero change on a truly flat name is
    rarer and less actionable than the false "market didn't move" this is
    guarding against on actively-trading commodities/crypto/futures.
    """
    if len(daily_close) < 2:
        return curr_price

    candidate = float(daily_close.iloc[-2])
    if abs(candidate - curr_price) < 0.001 and len(daily_close) >= 3:
        candidate = float(daily_close.iloc[-3])

    return candidate


def _trading_days_between(from_date, to_date) -> int:
    """Weekdays strictly after `from_date` through `to_date` inclusive.

    Same-day is 0, Friday->Monday is 1, Friday->Wednesday is 3. Market
    holidays count as trading days (there's no holiday calendar in this
    module), so this is a slight over-estimate across a holiday week — it
    is used only as a coarse staleness signal, where over-estimating means
    warning a day early rather than missing a genuinely stale bar.
    """
    if to_date <= from_date:
        return 0
    days = 0
    cursor = from_date
    while cursor < to_date:
        cursor += timedelta(days=1)
        if cursor.weekday() < 5:
            days += 1
    return days


# A US-hours daily bar older than this many trading days is the wrong price
# to be showing, whatever the hour. Two allows for yfinance's publishing lag
# on the current session plus one full prior session.
_MAX_DAILY_BAR_TRADING_DAYS = 2


def _get_current_price_and_bar_time(
    ticker: str,
    daily_close: pd.Series,
    intraday_close: pd.Series,
    naive_prior_close: float,
) -> tuple[float, object, bool]:
    """Current price, its source bar's timestamp, and whether to force-stale it.

    Prefers the most recent intraday bar; falls back to the latest daily
    bar. Returns `(curr_price, last_bar_time, force_stale)`.

    Intraday path keeps the long-standing sanity check: an intraday tick
    landing exactly on prior_close while today's own (still-forming) daily
    bar shows something different means the intraday bar hasn't updated
    yet, so the daily bar is trusted instead.

    Daily path adds an explicit currency check. Outside market hours the
    intraday fetch is skipped entirely for standard tickers, so the daily
    bar IS the price — and nothing downstream previously questioned how old
    it was, because the equity staleness threshold (75h, sized for a
    weekend gap) happily accepts a bar several sessions old. That is how a
    9 PM read could show Friday's close as if it were the current price
    after a Monday holiday: real bar, correctly labelled "Fri close", but
    presented as current with no warning. Anything past
    `_MAX_DAILY_BAR_TRADING_DAYS` trading days old now warns and is marked
    stale so callers render it as such.

    Crypto/futures are exempt from the trading-day check — they trade on
    their own calendar and have their own (much tighter) age thresholds in
    `_get_session_context()`.
    """
    if len(intraday_close) > 0:
        curr_price = float(intraday_close.iloc[-1])
        last_bar_time = intraday_close.index[-1]
        daily_last = float(daily_close.iloc[-1])
        if curr_price == naive_prior_close and daily_last != naive_prior_close:
            curr_price = daily_last
            last_bar_time = daily_close.index[-1]
        return curr_price, last_bar_time, False

    curr_price = float(daily_close.iloc[-1])
    last_bar_time = daily_close.index[-1]

    force_stale = False
    if ticker not in _EXTENDED_INTRADAY_SET:
        bar_ts = pd.Timestamp(last_bar_time)
        bar_utc = bar_ts.tz_localize("UTC") if bar_ts.tzinfo is None else bar_ts.tz_convert("UTC")
        now_et = datetime.now(pytz.utc).astimezone(pytz.timezone("America/New_York"))
        days_old = _trading_days_between(bar_utc.date(), now_et.date())
        if days_old > _MAX_DAILY_BAR_TRADING_DAYS:
            logger.warning(
                "PriceCache: %s falling back to a daily bar %d trading days old (%s) — "
                "yfinance has not published a recent session; marking stale",
                ticker, days_old, bar_utc.date().isoformat(),
            )
            force_stale = True

    return curr_price, last_bar_time, force_stale


def _get_session_context(
    ticker: str, last_bar_time, price: float, prev_close: float, force_stale: bool = False
) -> dict:
    """Session metadata for a price tick: how old it is, and a human-readable
    note on what it represents (a live intraday quote, a stale weekend
    close, etc.) — shown alongside the price in /prices and the brief so a
    number is never mistaken for fresher than it is.

    `last_bar_time` is whatever `refresh()` used as the source bar's
    timestamp (a pandas Timestamp).

    CRITICAL: yfinance daily bars are stamped 00:00 UTC of the trading day.
    Converting that straight to ET lands on 20:00 ET the PRIOR calendar day
    (e.g. Friday's close bar reads as "Thursday 20:00 ET") — every US-hours
    instrument's weekday label was off by one, and daily-bar age was being
    measured from that wrong instant. A bar stamped exactly at midnight UTC
    is detected as a daily bar and its UTC *date* (not an ET-converted
    datetime) is used as the trading date for display/age purposes.
    Intraday bars (5m during market hours, hourly fallback) carry a real
    time-of-day and still convert to ET normally.

    `session_label`/`condition_note` stay coarse ("14min ago", "Fri close
    (24/7)") for the internal staleness bucketing this function has always
    done. `exact_timestamp` is separate: always the precise timestamp
    (full date + ET time for an intraday bar, weekday + calendar date for
    a daily one) so a caller — bot.py's `_session_suffix()` — can show
    "exactly when" rather than only "roughly how long ago" (see Fix 1 /
    Fix 4 in the price_cache staleness work).

    `force_stale=True` marks the tick stale regardless of `last_bar_time`'s
    age, for callers that detected a problem the age threshold can't see.

    `price`/`prev_close` aren't currently used in the staleness/label
    computation itself (that's purely a function of `last_bar_time`) but
    are accepted for parity with the call site and in case a future
    refinement wants them (e.g. flagging a suspiciously unchanged price
    alongside an old timestamp).
    """
    now_utc = datetime.now(pytz.utc)
    now_et = now_utc.astimezone(pytz.timezone("America/New_York"))

    if last_bar_time is None:
        return {
            "last_update": None,
            "last_update_et": None,
            "data_age_hours": None,
            "data_lag_minutes": None,
            "exact_timestamp": None,
            "session_label": "unknown",
            "is_stale": True,
            "condition_note": "timestamp unavailable",
            "instrument_type": "unknown",
        }

    last_bar_time = pd.Timestamp(last_bar_time)
    last_bar_utc = last_bar_time.tz_localize("UTC") if last_bar_time.tzinfo is None else last_bar_time.tz_convert("UTC")

    is_daily_bar = last_bar_utc.hour == 0 and last_bar_utc.minute == 0 and last_bar_utc.second == 0

    et_tz = pytz.timezone("America/New_York")
    last_bar_et = last_bar_utc.astimezone(et_tz)

    if is_daily_bar:
        # Trading date = the UTC date of the bar itself, not the ET-shifted
        # datetime (which would land on the prior day). Age is measured in
        # whole calendar days from that trading date, since a daily bar has
        # no meaningful intraday precision to begin with.
        #
        # "Today" for that subtraction is the ET date, NOT the UTC date.
        # From 20:00 ET onward (19:00 in winter) the UTC date has already
        # rolled over, so comparing against it made every evening read of a
        # same-day close come back a full day older than it was: at 21:00 ET
        # Tuesday, Tuesday's own close reported "24h ago" instead of a fresh
        # same-session bar. A trading date is an ET concept and has to be
        # differenced against an ET date.
        trading_date = last_bar_utc.date()
        age_hours = (now_et.date() - trading_date).days * 24
        weekday_name = _WEEKDAY_NAMES[trading_date.weekday()]
        time_display = f"{weekday_name} close"
        exact_timestamp = f"{weekday_name} {trading_date.isoformat()} close"
        data_lag_minutes = None  # daily bar — age is meaningful in days, not minutes
    else:
        age_hours = (now_utc - last_bar_utc).total_seconds() / 3600
        weekday_name = _WEEKDAY_NAMES[last_bar_et.weekday()]
        time_display = f"{weekday_name} {last_bar_et.strftime('%H:%M ET')}"
        exact_timestamp = last_bar_et.strftime("%Y-%m-%d %H:%M ET")
        data_lag_minutes = round(age_hours * 60, 1)

    last_update_str = last_bar_utc.strftime("%Y-%m-%d %H:%M UTC")

    # Staleness thresholds — sized to cover a full weekend/market-closed gap
    # for each instrument type, not just an overnight one, since the prior
    # ~26-30h thresholds flagged every Monday-morning equity/index/FX read
    # as stale even though nothing was actually wrong with the data.
    if ticker in _CRYPTO_TICKER_SET:
        stale_threshold_hours, instrument_type = 4, "crypto"    # 24/7 — no weekend gap
    elif ticker in _FUTURES_TICKER_SET:
        stale_threshold_hours, instrument_type = 50, "futures"  # some futures trade Sat
    elif ticker in _INTL_TICKER_SET:
        stale_threshold_hours, instrument_type = 30, "intl_index"  # daily-session, unchanged
    elif ticker.startswith("^"):
        stale_threshold_hours, instrument_type = 75, "index"    # Fri close -> Mon open ~63h
    elif "=X" in ticker:
        stale_threshold_hours, instrument_type = 55, "fx"       # Fri 5pm ET -> Sun 5pm ET
    else:
        stale_threshold_hours, instrument_type = 75, "equity"   # Fri close -> Mon open ~63h

    # `force_stale` lets the caller override the age threshold when it has
    # evidence the age alone doesn't capture — specifically an after-hours
    # daily bar that's several trading days old for a US-hours instrument,
    # which is well inside the 75h weekend-tolerant equity threshold but
    # still the wrong price to show. See _get_current_price_and_bar_time().
    is_stale = force_stale or age_hours > stale_threshold_hours

    if age_hours < 0.5:
        session_label = "live"
    elif age_hours < 27:
        session_label = f"{int(age_hours)}h ago"
    else:
        session_label = f"{int(age_hours / 24)}d ago"
    if is_stale:
        # Driven by the actual `is_stale` verdict (per-instrument threshold)
        # rather than a fixed age bucket — otherwise a 3-day-old bar that's
        # still within a longer threshold (e.g. a Monday-morning equity read
        # after a weekend) would misleadingly say "STALE" anyway.
        session_label += " — STALE"

    if is_stale and weekday_name in ("Fri", "Sat", "Sun"):
        condition_note = f"Last: {time_display} — weekend close"
    elif instrument_type == "crypto":
        condition_note = f"Last: {time_display} (24/7)"
    elif instrument_type == "intl_index":
        condition_note = f"Last: {time_display} (local session)"
    else:
        condition_note = f"Last: {time_display}"

    return {
        "last_update": last_update_str,
        "last_update_et": last_bar_et.strftime("%Y-%m-%d %H:%M ET"),
        "data_age_hours": round(age_hours, 1),
        "data_lag_minutes": data_lag_minutes,
        "exact_timestamp": exact_timestamp,
        "session_label": session_label,
        "is_stale": is_stale,
        "condition_note": condition_note,
        "instrument_type": instrument_type,
    }


class PriceCache:
    """Thread-safe unified price cache.

    Use `get`/`get_prices`/`get_price`/`get_change_1d`/`get_yield` to read
    — never fetch yfinance (or FRED, for yields) directly in modules that
    only need a current price and its 1D change.
    """

    MARKET_HOURS_TTL = 15 * 60    # 15 minutes during market hours
    OFFHOURS_TTL = 60 * 60        # 60 minutes outside market hours

    def __init__(self):
        self._cache: dict[str, dict] = {}
        self._last_fetch: float = 0
        self._lock = threading.Lock()
        self._all_tickers: list[str] = []

    def _get_ttl(self) -> int:
        """Returns cache TTL based on market hours."""
        et = pytz.timezone("America/New_York")
        now_et = datetime.now(et)
        if now_et.weekday() >= 5:
            return self.OFFHOURS_TTL
        market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        if market_open <= now_et <= market_close:
            return self.MARKET_HOURS_TTL
        return self.OFFHOURS_TTL

    def _build_ticker_list(self) -> list[str]:
        """Builds the complete list of yfinance-batchable tickers to fetch.

        2Y/20Y Treasury (FRED-only) are NOT in this list — `refresh()`
        fetches those separately and merges them in under their FRED
        series id.
        """
        from equity.config.positions import POSITIONS, WATCHLIST

        tickers = set()
        tickers.update(POSITIONS.keys())
        tickers.update(WATCHLIST.keys())
        tickers.update(SECTOR_ETFS)
        tickers.update(BENCHMARK_ETFS)
        tickers.update(FACTOR_ETFS)
        tickers.update(TREASURY_PROXIES)
        tickers.update(FX_LIST)
        tickers.update(COMMODITY_LIST)
        tickers.update(EQUITY_FUTURES.keys())
        tickers.update(CRYPTO_TICKERS.keys())
        tickers.update(VOLATILITY_TICKERS.keys())
        tickers.update(INTERNATIONAL_INDICES.keys())
        tickers.update(CREDIT_TICKERS.keys())
        return sorted(tickers)

    def _fetch_fred_yields(self) -> dict[str, dict]:
        """2Y/20Y Treasury yields from FRED (not on yfinance — see module docstring).

        Never raises: a missing FRED_API_KEY, an unreachable API, or a
        malformed series just means those two entries are absent from the
        result, same as any other ticker `refresh()` fails to fetch.
        """
        if not FRED_API_KEY:
            logger.debug("PriceCache: FRED_API_KEY not set — 2Y/20Y Treasury unavailable")
            return {}

        try:
            from fredapi import Fred

            fred = Fred(api_key=FRED_API_KEY)
        except Exception as e:
            logger.warning("PriceCache: FRED client init failed: %s", e)
            return {}

        result: dict[str, dict] = {}
        for series_id, tenor in TREASURY_FRED_SERIES.items():
            try:
                series = fred.get_series(series_id).dropna()
                if len(series) < 1:
                    continue
                curr = float(series.iloc[-1])
                prev = float(series.iloc[-2]) if len(series) >= 2 else curr

                # FRED daily series carry a bare observation date, not a
                # timestamp — approximate its "as of" time as ~4-5pm ET
                # (21:00 UTC) that date, the usual publish time for these
                # series, for a staleness estimate. Precision beyond an
                # hour or two doesn't matter at the 75h threshold this is
                # checked against below (same weekend-gap-sized threshold
                # as equities in _get_session_context() — a 26h threshold
                # flagged every Monday's read as stale even though FRED
                # simply hadn't published a new observation over the
                # weekend, same underlying bug as the equity one).
                try:
                    obs_date = series.index[-1]
                    obs_utc = pd.Timestamp(obs_date.date()).tz_localize("UTC") + pd.Timedelta(hours=21)
                    age_hours = (datetime.now(pytz.utc) - obs_utc).total_seconds() / 3600
                    last_update = obs_date.strftime("%Y-%m-%d")
                    exact_timestamp = f"{_WEEKDAY_NAMES[obs_date.weekday()]} {last_update} close"
                except Exception:
                    age_hours, last_update, exact_timestamp = None, None, None

                result[series_id] = {
                    "price": curr,
                    "prev_close": prev,
                    "change_1d_bps": (curr - prev) * 100,
                    "as_of": datetime.now().isoformat(),
                    "is_yield": True,
                    "tenor": tenor,
                    "last_update": last_update,
                    "last_update_et": None,
                    "data_age_hours": round(age_hours, 1) if age_hours is not None else None,
                    "data_lag_minutes": None,  # daily series — no intraday lag concept
                    "exact_timestamp": exact_timestamp,
                    "session_label": "FRED daily",
                    "is_stale": (age_hours or 0) > 75,
                    "condition_note": "FRED daily publish ~3PM ET",
                    "instrument_type": "yield",
                }
            except Exception as e:
                logger.debug("PriceCache: FRED fetch failed for %s (%s): %s", series_id, tenor, e)
        return result

    def refresh(self, force: bool = False) -> bool:
        """Fetches fresh prices for all tickers.

        Returns True if a refresh occurred, False if the cache was still
        valid. Thread-safe.

        Two yfinance fetches during market hours (one daily, one intraday)
        rather than one: a 5d/1d bar for "today" is still forming during
        market hours, and often just shows the session open — especially
        for futures/commodities, whose overnight session can open exactly
        at the prior settlement, making an in-progress day look like a 0%
        move when the instrument has actually moved plenty intraday. The
        daily bars still supply prev_close (always `iloc[-2]`, the last
        *complete* bar — whether that's yesterday's during market hours or
        today's once the session's closed, "one bar back from the newest"
        is always the right anchor); the intraday bar supplies the actual
        current price while today's daily bar is still incomplete. Outside
        market hours the daily bar for "today" is already final, so the
        second fetch is skipped — it would just be discarded.

        The intraday fetch is 5-minute bars (`period="1d"` — yfinance only
        keeps 5m history for a handful of days, and a full day is all one
        refresh cycle needs) so the last completed bar is at most ~5min
        stale rather than up to an hour. Not every ticker carries 5m data
        (some international indices and thinly-traded ETFs don't) — those
        fall back to an hourly fetch (`intraday_fallback`) so they still
        get an intraday price during market hours rather than falling all
        the way back to yesterday's daily close.

        Crypto/futures/commodities (`_EXTENDED_INTRADAY_SET`) get a THIRD,
        separate intraday fetch — `intraday_extended` — that runs every
        refresh cycle regardless of `is_market_hours`, with a 5-day window
        rather than 1-day. Those instruments trade well outside NYSE cash
        hours, so gating them behind the NYSE check meant an off-hours or
        post-holiday refresh had nothing but a stale daily/settlement bar
        for gold, oil, ES futures, BTC, etc. even while they were actively
        trading. The wider 5-day window is a safety net against yfinance
        itself having a gap right at the edge of a 1-day window (e.g. the
        morning after a holiday).
        """
        ttl = self._get_ttl()
        is_market_hours = ttl == self.MARKET_HOURS_TTL
        now = time.time()

        with self._lock:
            if not force and (now - self._last_fetch) < ttl:
                return False

            tickers = self._build_ticker_list()

            try:
                daily = yf_download(
                    tickers, period="5d", interval="1d", auto_adjust=True, progress=False,
                )
                if daily.empty:
                    logger.warning("PriceCache.refresh: empty response from yfinance (daily)")
                    return False

                intraday_standard = None
                intraday_fallback = None
                if is_market_hours:
                    standard_tickers = [
                        t for t in tickers
                        if t not in _EXTENDED_INTRADAY_SET and t not in DAILY_ONLY_TICKERS
                    ]
                    intraday_standard = yf_download(
                        standard_tickers, period="1d", interval="5m", auto_adjust=True, progress=False,
                    ) if standard_tickers else pd.DataFrame()
                    if intraday_standard.empty:
                        intraday_standard = None
                    else:
                        # Yield tickers never use intraday data (see the
                        # `_YIELD_SCALE` branch below) — skip them here so
                        # the fallback fetch isn't wasted on data that will
                        # just be discarded.
                        missing_5m = [
                            t for t in standard_tickers
                            if t not in _YIELD_SCALE and len(_extract_close(intraday_standard, t)) == 0
                        ]
                        if missing_5m:
                            intraday_fallback = yf_download(
                                missing_5m, period="2d", interval="1h", auto_adjust=True, progress=False,
                            )
                            if intraday_fallback.empty:
                                intraday_fallback = None

                # Crypto/futures/commodities — always fetched, not gated by
                # is_market_hours. See refresh()'s docstring.
                extended_tickers = [
                    t for t in tickers
                    if t in _EXTENDED_INTRADAY_SET and t not in DAILY_ONLY_TICKERS
                ]
                intraday_extended = yf_download(
                    extended_tickers, period="5d", interval="5m", auto_adjust=True, progress=False,
                ) if extended_tickers else pd.DataFrame()
                if intraday_extended.empty:
                    intraday_extended = None

                new_cache: dict[str, dict] = {}
                for ticker in tickers:
                    try:
                        if isinstance(daily.columns, pd.MultiIndex):
                            if ("Close", ticker) not in daily.columns:
                                continue
                            daily_close = daily[("Close", ticker)].dropna()
                            volume = daily[("Volume", ticker)].dropna() if ("Volume", ticker) in daily.columns else pd.Series(dtype=float)
                        else:
                            daily_close = daily["Close"].dropna()
                            volume = daily["Volume"].dropna() if "Volume" in daily.columns else pd.Series(dtype=float)

                        if len(daily_close) < 1:
                            continue

                        # prev_close is always the second-to-last daily bar
                        # when there is one: during market hours iloc[-1] is
                        # today's still-forming bar (excluded); once the
                        # session's closed, iloc[-1] IS today's final close
                        # and iloc[-2] is yesterday's — still the right
                        # "one bar back" anchor either way. Only a ticker
                        # with a single bar of history at all (e.g. brand
                        # new listing) falls back to using that one bar as
                        # its own "prior close" (0% by construction).
                        #
                        # This naive version is only used below to sanity-
                        # check whether the intraday tick has actually
                        # updated — the real prior_close used for the
                        # cached change% (`_get_prior_close()`, after
                        # curr_price is settled) additionally guards against
                        # a stale/duplicate settlement bar at iloc[-2].
                        naive_prior_close = float(daily_close.iloc[-2]) if len(daily_close) >= 2 else float(daily_close.iloc[-1])

                        intraday_close = pd.Series(dtype=float)
                        if ticker not in _YIELD_SCALE:
                            if ticker in _EXTENDED_INTRADAY_SET:
                                intraday_close = _extract_close(intraday_extended, ticker)
                            else:
                                intraday_close = _extract_close(intraday_standard, ticker)
                                if len(intraday_close) == 0:
                                    intraday_close = _extract_close(intraday_fallback, ticker)

                        if ticker in _YIELD_SCALE:
                            # Yields: daily bars only. See this task's note
                            # — Treasury "intraday" bars aren't the fix
                            # target here (futures/commodities/crypto are),
                            # and FRED-sourced tenors (2Y/20Y, handled in
                            # _fetch_fred_yields()) don't have intraday data
                            # at all.
                            if len(daily_close) < 2:
                                continue
                            scale = _YIELD_SCALE[ticker]
                            curr_yield = float(daily_close.iloc[-1]) * scale
                            prev_yield = naive_prior_close * scale
                            session_ctx = _get_session_context(ticker, daily_close.index[-1], curr_yield, prev_yield)
                            session_ctx["instrument_type"] = "yield"
                            new_cache[ticker] = {
                                "price": curr_yield,
                                "prev_close": prev_yield,
                                "change_1d_bps": (curr_yield - prev_yield) * 100,
                                "as_of": datetime.now().isoformat(),
                                "is_yield": True,
                                "tenor": TREASURY_TICKERS.get(ticker),
                                **session_ctx,
                            }
                            continue

                        # Intraday-vs-daily selection, the intraday sanity
                        # check, and the after-hours daily-bar currency
                        # check all live in the helper — see
                        # _get_current_price_and_bar_time().
                        curr_price, last_bar_time, force_stale = _get_current_price_and_bar_time(
                            ticker, daily_close, intraday_close, naive_prior_close,
                        )

                        # Real prior_close for the cached change% — guards
                        # against a stale/duplicate settlement bar sitting
                        # at iloc[-2] (e.g. a thin-holiday commodity bar
                        # that just carries the prior session's value
                        # forward). See _get_prior_close().
                        prior_close = _get_prior_close(daily_close, curr_price)

                        change_1d_pct = (curr_price / prior_close - 1) * 100 if prior_close else 0.0
                        vol_today = float(volume.iloc[-1]) if len(volume) >= 1 else None
                        vol_avg = float(volume.iloc[-5:].mean()) if len(volume) >= 5 else None

                        # Exactly-0% on an instrument that's actively
                        # trading right now (crypto/futures/commodities —
                        # nights/weekends/holidays don't pause them) is more
                        # likely stale/duplicate intraday data than a
                        # genuine flat tick — worth a log line to catch a
                        # yfinance gap early rather than silently trusting it.
                        if change_1d_pct == 0.0 and ticker in _EXTENDED_INTRADAY_SET:
                            logger.warning(
                                "PriceCache: %s showing exactly 0%% change (curr=%s, prior=%s) — "
                                "likely stale intraday data",
                                ticker, curr_price, prior_close,
                            )

                        session_ctx = _get_session_context(
                            ticker, last_bar_time, curr_price, prior_close, force_stale=force_stale,
                        )
                        new_cache[ticker] = {
                            "price": curr_price,
                            "prev_close": prior_close,
                            "change_1d_pct": round(change_1d_pct, 3),
                            "volume_today": vol_today,
                            "volume_5d_avg": vol_avg,
                            "as_of": datetime.now().isoformat(),
                            "is_yield": False,
                            **session_ctx,
                        }
                    except Exception as e:
                        logger.debug("PriceCache: failed to process %s: %s", ticker, e)
                        continue

                new_cache.update(self._fetch_fred_yields())

                self._cache = new_cache
                self._all_tickers = tickers + list(TREASURY_FRED_SERIES.keys())
                self._last_fetch = now
                logger.info(
                    "PriceCache.refresh: %d/%d tickers updated (TTL=%dmin, market_hours=%s)",
                    len(new_cache), len(self._all_tickers), ttl // 60, is_market_hours,
                )
                return True

            except Exception as e:
                logger.error("PriceCache.refresh failed: %s", e)
                return False

    def get(self, ticker: str) -> Optional[dict]:
        """Price data for a single ticker (by its yfinance ticker or, for
        2Y/20Y, its FRED series id — see `get_yield()` for tenor lookups).
        Auto-refreshes if the cache is stale. None if unavailable.
        """
        self.refresh()
        return self._cache.get(ticker)

    def get_yield(self, tenor: str) -> Optional[dict]:
        """Price data for a Treasury tenor ('2Y', '5Y', '10Y', '20Y', '30Y', '3M'),
        regardless of whether it's sourced from yfinance or FRED.
        """
        cache_key = TENOR_TO_CACHE_KEY.get(tenor)
        if cache_key is None:
            return None
        return self.get(cache_key)

    def get_prices(self, tickers: list[str] | None = None) -> dict[str, dict]:
        """Price data for all tickers, or a subset if specified. Auto-refreshes if stale."""
        self.refresh()
        if tickers is None:
            return dict(self._cache)
        return {t: self._cache[t] for t in tickers if t in self._cache}

    def get_price(self, ticker: str) -> Optional[float]:
        """Convenience method — returns just the current price."""
        data = self.get(ticker)
        return data["price"] if data else None

    def get_change_1d(self, ticker: str) -> Optional[float]:
        """1D change: percent for a regular ticker, basis points for a yield.
        Check `data['is_yield']` (via `get()`) if the distinction matters —
        this just returns whichever field that ticker actually has.
        """
        data = self.get(ticker)
        if not data:
            return None
        return data.get("change_1d_bps") if data.get("is_yield") else data.get("change_1d_pct")

    def invalidate(self) -> None:
        """Forces the next get()/refresh() call to refetch. Thread-safe."""
        with self._lock:
            self._last_fetch = 0

    def coverage_report(self) -> str:
        """Summary of cache coverage for /status and /prices."""
        total = len(self._all_tickers)
        cached = len(self._cache)
        missing = [t for t in self._all_tickers if t not in self._cache]
        age_min = (time.time() - self._last_fetch) / 60 if self._last_fetch else float("inf")
        age_str = f"{age_min:.0f}min" if age_min != float("inf") else "never fetched"
        return (
            f"Price cache: {cached}/{total} tickers | "
            f"Age: {age_str} | "
            f'Missing: {missing[:5] if missing else "none"}'
        )


# Module-level singleton — import this everywhere.
price_cache = PriceCache()
