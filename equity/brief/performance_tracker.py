"""Daily performance snapshot: holdings vs benchmarks/sectors for the morning brief.

Portfolio P&L (`fetch_portfolio_performance()`) is a stub until IBKR
(Module 4) is connected — see `equity.portfolio.monitor.get_ibkr_positions()`
for the same stub pattern. Everything else here works immediately from
yfinance: broad-market/sector/factor benchmark returns, the SLV/GLD
risk-on/risk-off ratio, and each position's 1D return relative to its
mapped sector ETF (`market_config.POSITION_SECTOR_MAP`).

`fetch_benchmark_performance()` is one batched `yf.download()` call over
`market_config.BENCHMARK_TICKERS` — mirrors `equity.brief.market_snapshot`'s
batching approach — cached to `equity/data/cache/benchmark_performance.json`
for `BENCHMARK_CACHE_HOURS` hours (fetched_at + date + payload; a cache
written on a prior calendar date is always treated as stale regardless of
BENCHMARK_CACHE_HOURS, so an 11pm write never serves yesterday's prices
the next morning).

`fetch_position_relative_performance()` reuses that cached benchmark data
for each position's sector ETF (every `POSITION_SECTOR_MAP` value is itself
a `BENCHMARK_TICKERS` entry, so no second fetch is needed for the ETF side)
and only issues a second, small batch download for the position tickers
themselves.

`fetch_benchmark_performance()`'s batch also pulls `HG=F`/`GC=F`/`SI=F`
(COMEX copper/gold/silver futures) alongside `BENCHMARK_TICKERS` — none of
the three are `BENCHMARK_TICKERS` entries and none gets a display row of
its own; each price is only surfaced inline, as a reference alongside its
ETF: `$/lb` on the CPER line, `$/oz` on the GLD and SLV lines (see
`_format_cper_line()`/`_format_gld_line()`/`_format_slv_line()`).

These are front-month *futures* prices, not true spot — labeled as such
in the display (`(futures)`, not `(spot)`). FRED does not carry any
gold/silver spot or fixing series: this was originally specified to pull
LBMA fix data from FRED (`GOLDAMGBD228NLBM`/`SLVPRUSD`), but FRED pulled
that data ~2015 and never replaced it — confirmed empirically 2026-08-30
against the live FRED API (both series 400 "does not exist", and a
broader catalog search turns up no replacement covering precious-metals
spot prices). Front-month futures via yfinance are the standard live
substitute, same approach this module already used for copper via `HG=F`.

`fetch_spot_prices()` is a thin wrapper over `fetch_benchmark_performance()`
— no separate network call or cache; it just pulls the gold/silver futures
prices back out of that (1-hour-cached) batch. `format_performance_section()`
displays them alongside GLD/SLV's ETF price, falling back to ETF-only when
a futures price is unavailable.
"""

import json
import logging
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from equity.brief.market_snapshot import compute_ma_flags, format_flag
from equity.config import positions as positions_config
from equity.config.market_config import (
    BENCHMARK_TICKERS,
    GOLD_FUTURES_TICKER,
    POSITION_SECTOR_MAP,
    POSITION_UNDERPERFORM_ALERT_PCT,
    SILVER_FUTURES_TICKER,
    SILVER_GOLD_RATIO_RISK_ON_THRESHOLD,
    SLV_SI_DIVERGENCE_ALERT_PCT,
)
from equity.data.yfinance_utils import yf_download

logger = logging.getLogger(__name__)

_DIVIDER = "━━━━━━━━━━━━━━━━━━━━━━━━"

_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache"
_BENCHMARK_CACHE_PATH = _CACHE_DIR / "benchmark_performance.json"
BENCHMARK_CACHE_HOURS = 1

# Not a BENCHMARK_TICKERS entry — no display row of its own, only referenced
# inline on the CPER commodities line (see module docstring).
COPPER_FUTURES_TICKER = "HG=F"

# Display-only groupings/labels below — not thresholds, just how the
# formatter lays out BENCHMARK_TICKERS. Not sourced from market_config for
# the same reason market_snapshot.py's _RATE_LABELS/_COMMODITY_FORMATS
# aren't: they're presentation, not configuration.
_BROAD_MARKET = ["SPY", "QQQ", "IWM", "EEM", "EFA"]

_SECTOR_ROWS = [
    [("XLK", "Tech"), ("XLU", "Util")],
    [("XLF", "Fin"), ("XLV", "HC")],
    [("XLE", "Energy"), ("XLI", "Ind")],
    [("XLY", "Cyclical"), ("XLC", "Comm")],
    [("XLB", "Materials"), ("XLRE", "REIT")],
    [("TLT", "Gov Bond"), ("BIL", "Cash")],
]

_SPECIALIST = ["URA"]

# GLD/SLV/CPER get their own dedicated lines (spot/futures reference makes
# them too long to pair two-per-row) — see _format_gld_line()/_format_slv_line()
# /_format_cper_line(). This grid is just what's left: plain ETF price + 1D%.
_COMMODITY_ROWS = [
    [("TLT", "TLT", 1), ("SHY", "SHY", 1)],
    [("IEF", "IEF", 1)],
]

_SIGNAL_ARROWS = {"RISK_ON": "↑", "RISK_OFF": "↓", "NEUTRAL": "→"}

PORTFOLIO_STUB_FIELDS = [
    "portfolio_value", "change_1d_pct", "change_1d_abs",
    "change_1w_pct", "change_1m_pct", "change_3m_pct",
    "change_6m_pct", "change_ytd_pct", "change_1y_pct",
    "change_2y_pct", "change_3y_pct", "change_5y_pct",
    "cagr_1y", "cagr_3y", "cagr_5y",
    "cumulative_return_since_inception",
    "twr", "mwr",
    "sharpe_30d", "sharpe_90d", "sharpe_1y",
    "max_drawdown", "max_drawdown_duration_days",
    "win_rate_pct", "avg_winner_pct", "avg_loser_pct",
    "best_position", "worst_position",
]


# ---------------------------------------------------------------------------
# yfinance batch fetch
# ---------------------------------------------------------------------------

def _download_batch(tickers: list[str], period: str) -> pd.DataFrame | None:
    if not tickers:
        return None
    try:
        data = yf_download(
            tickers,
            period=period,
            auto_adjust=True,
            group_by="ticker",
            threads=True,
            progress=False,
        )
    except Exception as exc:  # yfinance can raise a variety of things on network/format issues
        logger.warning("yf.download failed for batch (period=%s): %s", period, exc)
        return None
    return None if data.empty else data


def _ticker_close(data: pd.DataFrame | None, ticker: str) -> pd.Series | None:
    """Close-price series for `ticker` from a multi-ticker yf.download() result, or None."""
    if data is None:
        return None
    try:
        if isinstance(data.columns, pd.MultiIndex):
            if ticker not in data.columns.get_level_values(0):
                return None
            sub = data[ticker]
        else:
            sub = data
        if "Close" not in sub.columns:
            return None
        close = sub["Close"].dropna()
        return close if not close.empty else None
    except Exception as exc:
        logger.warning("Could not extract close series for %s: %s", ticker, exc)
        return None


def _pct_change(close: pd.Series, idx: int) -> float | None:
    """% change from `close`'s row at `idx` (negative = from the end, 0 = first row) to the latest close.

    None if `idx` falls outside the available history (e.g. change_1m_pct
    early in the year, before 21 trading days have accumulated) rather than
    raising — "not enough history yet" is a normal, expected state here.
    """
    n = len(close)
    pos = idx if idx >= 0 else n + idx
    if pos < 0 or pos >= n:
        return None
    base = float(close.iloc[pos])
    if not base:
        return None
    return (float(close.iloc[-1]) / base - 1) * 100


def _price_years_ago(close: pd.Series, years: float) -> float | None:
    """Price closest to `years` calendar years before `close`'s last date, or None if not enough history."""
    idx = close.index
    target = idx[-1] - pd.Timedelta(days=int(365 * years))
    if target < idx[0]:
        return None  # series doesn't actually go back that far
    pos = idx.searchsorted(target)
    pos = min(max(int(pos), 0), len(idx) - 1)
    base = float(close.iloc[pos])
    return base if base else None


def _cagr(close: pd.Series, years: float) -> float | None:
    """Annualized return over `years`, or None if `close` doesn't have `years` of history."""
    base = _price_years_ago(close, years)
    if base is None:
        return None
    return ((float(close.iloc[-1]) / base) ** (1 / years) - 1) * 100


# ---------------------------------------------------------------------------
# Benchmark cache (1 hour)
# ---------------------------------------------------------------------------

def _load_benchmark_cache() -> dict | None:
    if not _BENCHMARK_CACHE_PATH.exists():
        return None
    try:
        with open(_BENCHMARK_CACHE_PATH) as f:
            cache = json.load(f)
        fetched_at = datetime.fromisoformat(cache["fetched_at"])
        cached_date = cache.get("date")
        payload = cache["data"]
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.warning("benchmark_performance.json cache unreadable, ignoring: %s", exc)
        return None

    if cached_date != date.today().isoformat():
        return None  # written on a prior trading day — force refresh regardless of BENCHMARK_CACHE_HOURS
    if (datetime.now() - fetched_at).total_seconds() > BENCHMARK_CACHE_HOURS * 3600:
        return None
    return payload


def _write_benchmark_cache(data: dict) -> None:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(_BENCHMARK_CACHE_PATH, "w") as f:
            json.dump(
                {"fetched_at": datetime.now().isoformat(), "date": date.today().isoformat(), "data": data},
                f, indent=2,
            )
    except OSError as exc:
        logger.warning("Failed to write benchmark_performance.json cache: %s", exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_benchmark_performance() -> dict:
    """Batch-fetch every `market_config.BENCHMARK_TICKERS`, cache-first (1 hour).

    Fetches 5 years of daily history (up from 'ytd') so 1D/1W/1M/1Y windows
    plus 3Y/5Y annualized CAGR (`market_config.PERFORMANCE_PERIODS`) and
    `compute_ma_flags()` (SMA20/50/200 proximity, 5Y high/low) are all
    computable from one batch.

    Never raises: a failed batch download or a ticker missing enough
    history for a given window is recorded in `data_warnings` and that
    field is left None rather than crashing the whole fetch.
    """
    cached = _load_benchmark_cache()
    if cached is not None:
        return cached

    data_warnings: list[str] = []
    tickers = list(BENCHMARK_TICKERS) + [COPPER_FUTURES_TICKER, GOLD_FUTURES_TICKER, SILVER_FUTURES_TICKER]
    data = _download_batch(tickers, period="5y")
    if data is None:
        data_warnings.append("yf.download returned no data for the benchmark batch")

    benchmarks: dict[str, dict] = {}
    for ticker in BENCHMARK_TICKERS:
        close = _ticker_close(data, ticker)
        if close is None or len(close) < 2:
            data_warnings.append(f"{ticker}: price data unavailable")
            continue

        price = float(close.iloc[-1])
        # 1Y is a plain (non-annualized) return; 3Y/5Y are annualized CAGR
        # per market_config.PERFORMANCE_PERIODS — years=1 would make
        # _cagr()'s **(1/1) a no-op anyway, but _price_years_ago() directly
        # makes that "plain return, not CAGR" distinction explicit.
        price_1y_ago = _price_years_ago(close, 1)
        entry = {
            "price": price,
            "change_1d_pct": _pct_change(close, -2),
            "change_1w_pct": _pct_change(close, -6),
            "change_1m_pct": _pct_change(close, -22),
            "change_1y_pct": (price / price_1y_ago - 1) * 100 if price_1y_ago else None,
            "cagr_3y": _cagr(close, 3),
            "cagr_5y": _cagr(close, 5),
            "ma_flags": compute_ma_flags(close, price),
        }

        if entry["change_1w_pct"] is None:
            data_warnings.append(f"{ticker}: fewer than 5 trading days — change_1w_pct unavailable")
        if entry["change_1m_pct"] is None:
            data_warnings.append(f"{ticker}: fewer than 21 trading days — change_1m_pct unavailable")
        benchmarks[ticker] = entry

    slv_close = _ticker_close(data, "SLV")
    gld_close = _ticker_close(data, "GLD")
    slv_gld_ratio = slv_gld_ratio_1d_change = None
    slv_gld_signal = "NEUTRAL"
    if slv_close is not None and gld_close is not None and len(slv_close) >= 2 and len(gld_close) >= 2:
        ratio_today = float(slv_close.iloc[-1]) / float(gld_close.iloc[-1])
        ratio_yesterday = float(slv_close.iloc[-2]) / float(gld_close.iloc[-2])
        slv_gld_ratio = ratio_today
        if ratio_yesterday:
            slv_gld_ratio_1d_change = (ratio_today / ratio_yesterday - 1) * 100
            if abs(slv_gld_ratio_1d_change) < 0.1:
                slv_gld_signal = "NEUTRAL"
            elif slv_gld_ratio_1d_change > SILVER_GOLD_RATIO_RISK_ON_THRESHOLD:
                slv_gld_signal = "RISK_ON"
            else:
                slv_gld_signal = "RISK_OFF"
    else:
        data_warnings.append("SLV/GLD ratio unavailable — missing SLV or GLD price data")

    # HG=F/GC=F/SI=F (COMEX copper/gold/silver futures) — none are
    # BENCHMARK_TICKERS entries, so they're pulled out of the batch
    # separately; only their latest prices are used, as $/lb or $/oz
    # references alongside CPER/GLD/SLV's ETF price.
    def _futures_price(ticker: str) -> float | None:
        close = _ticker_close(data, ticker)
        if close is None or len(close) < 1:
            data_warnings.append(f"{ticker}: price data unavailable")
            return None
        return float(close.iloc[-1])

    copper_futures_price = _futures_price(COPPER_FUTURES_TICKER)
    gold_futures_price = _futures_price(GOLD_FUTURES_TICKER)
    silver_futures_price = _futures_price(SILVER_FUTURES_TICKER)

    result = {
        "benchmarks": benchmarks,
        "copper_futures_price": copper_futures_price,
        "gold_futures_price": gold_futures_price,
        "silver_futures_price": silver_futures_price,
        "slv_gld_ratio": slv_gld_ratio,
        "slv_gld_ratio_1d_change": slv_gld_ratio_1d_change,
        "slv_gld_signal": slv_gld_signal,
        "as_of": datetime.now().isoformat(),
        "data_warnings": data_warnings,
    }
    _write_benchmark_cache(result)
    return result


def fetch_portfolio_performance() -> dict:
    """IBKR portfolio P&L stub (Module 4 — not yet connected).

    Once IBKR's Flex API is wired up, this should pull real fills/NAV
    history and populate every field in `PORTFOLIO_STUB_FIELDS`; callers
    should branch on `ibkr_connected` rather than assuming these fields
    exist.
    """
    return {
        "ibkr_connected": False,
        "message": "Connect IBKR (Module 4) to enable portfolio performance tracking.",
        "stub_fields": PORTFOLIO_STUB_FIELDS,
    }


def fetch_spot_prices(benchmark_data: dict | None = None) -> dict:
    """Gold/silver futures reference prices (GC=F/SI=F), pulled from `fetch_benchmark_performance()`.

    No separate network call or cache — `fetch_benchmark_performance()`
    already pulls both tickers into its batch (see module docstring for
    why these are futures, not FRED spot). Field names keep the
    `_spot_usd` naming `format_performance_section()`'s callers already use;
    the values themselves are front-month futures prices, displayed as
    such (`(futures)`, not `(spot)`) — see `_format_gld_line()`/
    `_format_slv_line()`. Never raises: a price missing from the batch just
    comes back None, and callers fall back to ETF-only display.
    """
    if benchmark_data is None:
        benchmark_data = fetch_benchmark_performance()
    return {
        "gold_spot_usd": benchmark_data.get("gold_futures_price"),
        "silver_spot_usd": benchmark_data.get("silver_futures_price"),
        "as_of": benchmark_data.get("as_of", datetime.now().isoformat()),
    }


def fetch_position_relative_performance(benchmark_data: dict | None = None) -> dict:
    """Each `positions.POSITIONS` ticker's 1D return vs its mapped sector ETF, plus its own MA/extremes flags.

    Reuses `benchmark_data` (fetching it fresh via
    `fetch_benchmark_performance()` if not supplied) for the sector-ETF side
    of the comparison — every `POSITION_SECTOR_MAP` value is itself a
    `BENCHMARK_TICKERS` entry. A position with no `POSITION_SECTOR_MAP`
    entry is skipped (logged, not an error); a position whose own price
    fetch fails, or whose sector ETF has no benchmark data, comes back with
    an `'error'` key instead of raising.

    Fetches 5 years of daily history per position (up from 5 days) so
    `compute_ma_flags()` can flag SMA20/50/200 proximity and 5Y high/low —
    surfaced as `position_flags` on each result.
    """
    if benchmark_data is None:
        benchmark_data = fetch_benchmark_performance()
    benchmarks = benchmark_data.get("benchmarks", {})

    tickers = list(positions_config.POSITIONS)
    data = _download_batch(tickers, period="5y")

    result: dict[str, dict] = {}
    for ticker in tickers:
        sector_etf = POSITION_SECTOR_MAP.get(ticker)
        if sector_etf is None:
            logger.warning("No POSITION_SECTOR_MAP entry for %s — skipping relative performance", ticker)
            continue

        close = _ticker_close(data, ticker)
        if close is None or len(close) < 2:
            result[ticker] = {"sector_etf": sector_etf, "error": "price data unavailable", "position_flags": []}
            continue
        position_change_1d_pct = _pct_change(close, -2)
        position_flags = compute_ma_flags(close, float(close.iloc[-1]))

        sector_entry = benchmarks.get(sector_etf)
        if sector_entry is None or sector_entry.get("change_1d_pct") is None:
            result[ticker] = {
                "sector_etf": sector_etf,
                "position_change_1d_pct": position_change_1d_pct,
                "error": f"{sector_etf} price data unavailable",
                "position_flags": position_flags,
            }
            continue

        sector_etf_change_1d_pct = sector_entry["change_1d_pct"]
        relative_performance_1d_pct = position_change_1d_pct - sector_etf_change_1d_pct
        result[ticker] = {
            "sector_etf": sector_etf,
            "position_change_1d_pct": position_change_1d_pct,
            "sector_etf_change_1d_pct": sector_etf_change_1d_pct,
            "relative_performance_1d_pct": relative_performance_1d_pct,
            "outperforming": relative_performance_1d_pct > 0,
            "position_flags": position_flags,
        }

    return result


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _fmt_pct(value: float | None) -> str:
    return f"{value:+.1f}%" if value is not None else "n/a"


def _fmt_amount(value: float | None, decimals: int) -> str:
    return f"${value:,.{decimals}f}" if value is not None else "n/a"


def _fmt_multi_period(b: dict) -> str:
    """1D / 1W / 1M / 1Y / 5Y-ann returns, space-separated — the common row shape for factor lines."""
    return (
        f"{_fmt_pct(b.get('change_1d_pct')):>7} {_fmt_pct(b.get('change_1w_pct')):>7} "
        f"{_fmt_pct(b.get('change_1m_pct')):>7} {_fmt_pct(b.get('change_1y_pct')):>7} "
        f"{_fmt_pct(b.get('cagr_5y')):>7}"
    )


def _format_gld_line(b: dict | None, gold_futures_price: float | None) -> str:
    """GLD's ETF returns (1D/1W/1M/1Y/5Yann), plus GC=F front-month futures price alongside.

    Labeled '(futures)', not '(spot)' — GC=F is a front-month futures
    price, not true spot (see module docstring on why FRED spot isn't
    available). GLD holds ~0.1 troy oz of gold per share, hence the
    '~0.1oz' note — that's why the ETF price runs ~10x below the futures
    price, not a data problem (contrast SLV, which holds ~1oz/share and
    so should track its futures price closely — see `_format_slv_line()`).
    """
    if b is None:
        return "GLD (ETF ~0.1oz)  data unavailable"
    line = f"GLD (ETF ~0.1oz) {_fmt_multi_period(b)}"
    if gold_futures_price is not None:
        line += f"  | GC=F {_fmt_amount(gold_futures_price, 0)}/oz"
    return line + format_flag(b.get("ma_flags", []))


def _format_slv_line(b: dict | None, silver_futures_price: float | None) -> str:
    """SLV's ETF returns (1D/1W/1M/1Y/5Yann), plus SI=F front-month futures price alongside.

    Labeled '(futures)', not '(spot)' — see `_format_gld_line()`. SLV holds
    ~1 troy oz of silver per share (vs. GLD's ~0.1oz), so unlike the GLD/
    GC=F pair these two prices should land close together; a gap past
    `market_config.SLV_SI_DIVERGENCE_ALERT_PCT` gets a ⚠️ flag rather than
    being displayed as if it were normal.
    """
    if b is None:
        return "SLV (ETF ~1oz)  data unavailable"
    line = f"SLV (ETF ~1oz)  {_fmt_multi_period(b)}"
    if silver_futures_price is not None:
        line += f"  | SI=F {_fmt_amount(silver_futures_price, 1)}/oz"
        if b.get("price") is not None and silver_futures_price:
            divergence_pct = abs(b["price"] - silver_futures_price) / silver_futures_price * 100
            if divergence_pct > SLV_SI_DIVERGENCE_ALERT_PCT:
                line += f" ⚠️ {divergence_pct:.1f}% divergence"
    return line + format_flag(b.get("ma_flags", []))


def _format_cper_line(b: dict | None, copper_futures_price: float | None) -> str:
    """CPER's ETF returns (1D/1W/1M/1Y/5Yann), plus HG=F (COMEX copper futures, $/lb) as a reference.

    CPER keeps its plain ETF label throughout (no '(ETF)' annotation like
    GLD/SLV) — see module docstring: copper spot is adequately represented
    by the COMEX futures price, this is a reference, not a fallback pair.
    """
    if b is None:
        return "CPER  data unavailable"
    line = f"CPER  {_fmt_multi_period(b)}"
    if copper_futures_price is not None:
        line += f"  | HG=F {_fmt_amount(copper_futures_price, 2)}/lb"
    return line + format_flag(b.get("ma_flags", []))


def _relative_label(rel: float) -> tuple[str, str]:
    """(verb, marker) for a position's return relative to its sector ETF.

    Marker is ⚠️ only past `market_config.POSITION_UNDERPERFORM_ALERT_PCT`
    (a >2% single-day underperformance) — a smaller shortfall is a
    "slight underperform" with no alert marker.
    """
    if rel > 0:
        return "outperform", "✓"
    if rel <= -POSITION_UNDERPERFORM_ALERT_PCT:
        return "underperform", "⚠️"
    return "slight underperform", ""


def _sector_movers(benchmarks: dict, n: int = 2) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """Top-`n` and bottom-`n` sector ETFs by 1D% among `_SECTOR_ROWS` tickers."""
    sector_tickers = [ticker for row in _SECTOR_ROWS for ticker, _ in row]
    movers = [
        (ticker, benchmarks[ticker]["change_1d_pct"])
        for ticker in sector_tickers
        if benchmarks.get(ticker) and benchmarks[ticker].get("change_1d_pct") is not None
    ]
    ranked = sorted(movers, key=lambda pair: pair[1], reverse=True)
    return ranked[:n], list(reversed(ranked[-n:])) if ranked else []


def _sector_short_label(ticker: str) -> str:
    for row in _SECTOR_ROWS:
        for t, label in row:
            if t == ticker:
                return label
    return ticker


def _build_highlights(benchmark_data: dict, relative_data: dict) -> list[str]:
    highlights = []
    benchmarks = benchmark_data.get("benchmarks", {})

    top, worst = _sector_movers(benchmarks)
    if top:
        ticker, pct = top[0]
        highlights.append(f"Sector top: {ticker} {_sector_short_label(ticker)} {_fmt_pct(pct)}")
    if worst:
        ticker, pct = worst[0]
        highlights.append(f"Sector worst: {ticker} {_sector_short_label(ticker)} {_fmt_pct(pct)}")

    divergences = [
        (ticker, entry["relative_performance_1d_pct"])
        for ticker, entry in relative_data.items()
        if "relative_performance_1d_pct" in entry and abs(entry["relative_performance_1d_pct"]) > POSITION_UNDERPERFORM_ALERT_PCT
    ]
    if divergences:
        for ticker, rel in sorted(divergences, key=lambda pair: abs(pair[1]), reverse=True):
            highlights.append(f"Position divergence >{POSITION_UNDERPERFORM_ALERT_PCT:.0f}%: {ticker} {_fmt_pct(rel)}")
    else:
        highlights.append(f"Position divergence >{POSITION_UNDERPERFORM_ALERT_PCT:.0f}%: none today")

    for ticker, entry in relative_data.items():
        for flag in entry.get("position_flags") or []:
            if "5Y" in flag:  # only the extremes flags, not every MA-proximity flag, rise to HIGHLIGHTS
                highlights.append(f"{ticker} {flag} — review thesis")

    for ticker in _SPECIALIST:
        b = benchmarks.get(ticker)
        if not b:
            continue
        for flag in b.get("ma_flags") or []:
            if "5Y" in flag:
                highlights.append(f"{ticker} {flag} — {BENCHMARK_TICKERS.get(ticker, ticker)} sector weak/strong")

    signal = benchmark_data.get("slv_gld_signal")
    if signal:
        highlights.append(f"SLV/GLD signal: {signal}")

    return highlights


def format_performance_section(benchmark_data: dict, portfolio_data: dict, relative_data: dict, spot_data: dict | None = None) -> str:
    """Render the `fetch_*` dicts above as one Telegram-ready string. Never raises.

    `spot_data` (from `fetch_spot_prices()`) is optional — omitted or with
    both prices None, GLD/SLV just render ETF-only (see
    `_format_gld_line()`/`_format_slv_line()`).
    """
    lines = ["📈 PERFORMANCE", _DIVIDER]

    if portfolio_data.get("ibkr_connected"):
        value = portfolio_data.get("portfolio_value")
        change = portfolio_data.get("change_1d_pct")
        portfolio_line = f"Portfolio: {_fmt_amount(value, 0)}  {_fmt_pct(change)} today" if value is not None else "Portfolio: data unavailable"
    else:
        portfolio_line = f"Portfolio: {portfolio_data.get('message', 'Connect IBKR for P&L tracking.')}"
    lines.append(portfolio_line)
    lines.append("")

    benchmarks = benchmark_data.get("benchmarks", {})

    lines.append("Benchmarks (1D / 1W / 1M / 1Y / 3Y ann / 5Y ann)")
    for ticker in _BROAD_MARKET:
        b = benchmarks.get(ticker)
        if b is None:
            lines.append(f"{ticker:<6} data unavailable")
            continue
        lines.append(
            f"{ticker:<6} {_fmt_pct(b['change_1d_pct']):>7} {_fmt_pct(b['change_1w_pct']):>7} "
            f"{_fmt_pct(b['change_1m_pct']):>7} {_fmt_pct(b['change_1y_pct']):>7} "
            f"{_fmt_pct(b.get('cagr_3y')):>7} {_fmt_pct(b.get('cagr_5y')):>7}"
            f"{format_flag(b.get('ma_flags', []))}"
        )
    lines.append("")

    lines.append("Sectors (1D / 1W / 1M / 1Y)")
    for row in _SECTOR_ROWS:
        for ticker, label in row:
            b = benchmarks.get(ticker)
            if b is None:
                lines.append(f"{ticker} {label:<10} data unavailable")
                continue
            lines.append(
                f"{ticker} {label:<10}{_fmt_pct(b['change_1d_pct']):>7} {_fmt_pct(b['change_1w_pct']):>7} "
                f"{_fmt_pct(b['change_1m_pct']):>7} {_fmt_pct(b['change_1y_pct']):>7}"
                f"{format_flag(b.get('ma_flags', []))}"
            )
    lines.append("")

    top, worst = _sector_movers(benchmarks)
    if top or worst:
        lines.append("Sector Highlights")
        if top:
            lines.append("  🟢 Top:   " + "  |  ".join(f"{t} {_sector_short_label(t)} {_fmt_pct(p)}" for t, p in top))
        if worst:
            lines.append("  🔴 Worst: " + "  |  ".join(f"{t} {_sector_short_label(t)} {_fmt_pct(p)}" for t, p in worst))
        lines.append("")

    lines.append("Commodities & Factors (1D / 1W / 1M / 1Y / 5Y ann)")
    spot_data = spot_data or {}
    lines.append(_format_gld_line(benchmarks.get("GLD"), spot_data.get("gold_spot_usd")))
    lines.append(_format_slv_line(benchmarks.get("SLV"), spot_data.get("silver_spot_usd")))
    lines.append(_format_cper_line(benchmarks.get("CPER"), benchmark_data.get("copper_futures_price")))

    signal = benchmark_data.get("slv_gld_signal")
    arrow = _SIGNAL_ARROWS.get(signal, "")
    lines.append(f"SLV/GLD ratio: {arrow} {signal or 'n/a'}")

    for ticker in _SPECIALIST:
        b = benchmarks.get(ticker)
        if b is None:
            lines.append(f"{ticker}  data unavailable")
            continue
        relevant = sorted(pos for pos, etf in POSITION_SECTOR_MAP.items() if etf == ticker)
        line = f"{ticker:<5} {_fmt_multi_period(b)}{format_flag(b.get('ma_flags', []))}"
        if relevant:
            line += f"  ← relevant to {', '.join(relevant)}"
        lines.append(line)

    for row in _COMMODITY_ROWS:
        for ticker, label, _decimals in row:
            b = benchmarks.get(ticker)
            if b is None:
                lines.append(f"{label:<5} data unavailable")
                continue
            lines.append(f"{label:<5} {_fmt_multi_period(b)}{format_flag(b.get('ma_flags', []))}")
    lines.append("")

    lines.append("Your Positions vs Sector (1D)")
    if relative_data:
        for ticker, entry in relative_data.items():
            flag_str = format_flag(entry.get("position_flags") or [])
            if "relative_performance_1d_pct" not in entry:
                lines.append(f"{ticker:<5} {entry.get('error', 'data unavailable')}{flag_str}")
                continue
            rel = entry["relative_performance_1d_pct"]
            verb, marker = _relative_label(rel)
            suffix = f" {marker}" if marker else ""
            lines.append(
                f"{ticker:<5} {_fmt_pct(entry['position_change_1d_pct'])}  vs {entry['sector_etf']:<4} "
                f"{_fmt_pct(entry['sector_etf_change_1d_pct'])}  → {_fmt_pct(rel)} {verb}{suffix}{flag_str}"
            )
    else:
        lines.append("(no positions)")

    highlights = _build_highlights(benchmark_data, relative_data)
    if highlights:
        lines.append("")
        lines.append("HIGHLIGHTS")
        lines.extend(f"• {h}" for h in highlights)

    warnings = benchmark_data.get("data_warnings") or []
    if warnings:
        lines.append("")
        lines.extend(f"⚠️ {w}" for w in warnings)

    lines.append(_DIVIDER)
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    benchmark_result = fetch_benchmark_performance()
    portfolio_result = fetch_portfolio_performance()
    relative_result = fetch_position_relative_performance(benchmark_result)
    spot_result = fetch_spot_prices(benchmark_result)

    print(format_performance_section(benchmark_result, portfolio_result, relative_result, spot_result))
