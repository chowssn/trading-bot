"""Cheap price-only pre-filter for the equity screener.

Runs first, on every Russell 1000 name, before any paid FMP fundamental
calls — yfinance only, batched via `yf.download()` to keep it fast. A
ticker "passes" if its price action looks like a dislocation worth
investigating further (down meaningfully over the last year, RSI showing
sustained weakness, and liquid enough to actually trade), per the
thresholds in `equity.config.settings`.

This is a technical screen only — it says nothing about whether the
weakness is justified by fundamentals. That's the next stage.

Market cap filtering also happens here rather than in `universe.py`: the
IWB CSV's `market_value` column is the fund's dollar position in a name,
not the company's market cap, so it can't be used as a market cap floor
(see `universe.py` module docstring). Instead, once a ticker has passed
the price/RSI/volume dislocation filter, we fetch its real market cap from
`yf.Ticker(ticker).info` — one network call per name, only affordable
because by this point the candidate set is small (typically 30-80 names,
not the full ~1000-ticker universe).
"""

import logging
import time
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

from backtest.indicators import rsi
from equity.config import settings
from equity.config.market_config import (
    RETURN_3M_GRADUAL_GRIND,
    RETURN_3M_SHARP_DROP,
    RSI_30D_THRESHOLD_BY_REGIME,
)
from equity.data.yfinance_utils import yf_download
from equity.screener.timing_signal import classify_timing

logger = logging.getLogger(__name__)

CACHE_DIR = Path(settings.CACHE_DIR)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CACHE_TTL_SECONDS = settings.PRICE_CACHE_HOURS * 60 * 60

YF_DOWNLOAD_PERIOD = "2y"
# ~1 trading year for the return_1y lookback, plus RSI/direction warmup.
MIN_TRADING_DAYS = 260

OUTPUT_COLUMNS = [
    "ticker",
    "return_1y",
    "return_3m",
    "price_action_type",
    "rsi_30d",
    "rsi_14d",
    "rsi_14d_direction",
    "avg_volume_30d",
    "passes_dislocation",
    "market_cap_b",
    "market_cap_unverified",
    "timing_signal",
    "timing_note",
]


def _is_cache_fresh(path: Path) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < CACHE_TTL_SECONDS


def _cache_path(as_of: date | None = None) -> Path:
    as_of = as_of or date.today()
    return CACHE_DIR / f"price_filter_{as_of.isoformat()}.csv"


def _chunk(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _extract_ticker_frame(data: pd.DataFrame, ticker: str, single_ticker: bool) -> pd.DataFrame:
    """Pull `ticker`'s OHLCV sub-frame out of a (possibly multi-ticker) yf.download() result."""
    if isinstance(data.columns, pd.MultiIndex):
        if ticker not in data.columns.get_level_values(0):
            raise ValueError("no data returned")
        sub = data[ticker]
    else:
        # A batch of exactly one ticker downloads without MultiIndex columns.
        if not single_ticker:
            raise ValueError("expected multi-level columns for a multi-ticker batch")
        sub = data

    if "Close" not in sub.columns:
        raise ValueError("missing Close column")
    sub = sub.dropna(subset=["Close"])
    if sub.empty:
        raise ValueError("empty price series")
    return sub


def _get_rsi_threshold(regime_flags: list[str] | None) -> float:
    """Active RSI 30D dislocation threshold for today's regime.

    `settings.RSI_30D_MAX` is the fallback when `regime_flags` is empty/None
    (regime unknown or not yet classified) — everything else comes from
    `market_config.RSI_30D_THRESHOLD_BY_REGIME`. HIGH_VOL / ELEVATED_VOL
    take priority over a RISK_ON_DAY flag (a volatile risk-on day is still
    a volatile day), and a RISK_ON_DAY only relaxes the bar when there's no
    offsetting RISK_OFF_DAY flag active the same day.
    """
    if not regime_flags:
        return settings.RSI_30D_MAX
    if "HIGH_VOL" in regime_flags:
        return RSI_30D_THRESHOLD_BY_REGIME["HIGH_VOL"]
    if "ELEVATED_VOL" in regime_flags:
        return RSI_30D_THRESHOLD_BY_REGIME["ELEVATED_VOL"]
    if "RISK_ON_DAY" in regime_flags and "RISK_OFF_DAY" not in regime_flags:
        return RSI_30D_THRESHOLD_BY_REGIME["RISK_ON"]
    return RSI_30D_THRESHOLD_BY_REGIME["NEUTRAL"]


def _classify_price_action(return_1y: float, return_3m: float) -> str:
    """Classifies the nature of the price dislocation.

    SHARP_RECENT:  3M drop > RETURN_3M_SHARP_DROP — recent sharp selloff
    SLOW_GRIND:    1Y bad but 3M only slightly negative — sustained weakness
    RECENT_BOUNCE: 1Y bad but 3M positive — may be recovering, watch RSI
    MIXED:         everything else
    """
    if return_3m < RETURN_3M_SHARP_DROP:
        return "SHARP_RECENT"
    if return_1y < settings.DISLOCATION_ZONE_MIN and return_3m > RETURN_3M_GRADUAL_GRIND:
        return "SLOW_GRIND"
    if return_1y < settings.DISLOCATION_ZONE_MIN and return_3m > 0:
        return "RECENT_BOUNCE"
    return "MIXED"


def _compute_row(ticker: str, sub: pd.DataFrame, rsi_threshold: float) -> dict:
    """Compute the price-filter row for one ticker's OHLCV frame, or raise ValueError."""
    if "Volume" not in sub.columns:
        raise ValueError("missing Volume column")

    close = sub["Close"].dropna()
    volume = sub["Volume"].dropna()
    if len(close) < MIN_TRADING_DAYS:
        raise ValueError(f"only {len(close)} trading days of history, need >= {MIN_TRADING_DAYS}")

    # return_1y: price closest to 1 calendar year ago vs. today.
    idx = close.index
    target_date = idx[-1] - pd.Timedelta(days=365)
    pos = idx.searchsorted(target_date)
    pos = min(max(int(pos), 0), len(idx) - 1)
    price_1y_ago = close.iloc[pos]
    price_today = close.iloc[-1]
    if not price_1y_ago or pd.isna(price_1y_ago) or pd.isna(price_today):
        raise ValueError("invalid price for return_1y calc")
    return_1y = (price_today / price_1y_ago - 1) * 100

    # return_3m: price closest to 3 calendar months ago vs. today.
    three_months_ago = idx[-1] - pd.DateOffset(months=3)
    idx_3m = idx.searchsorted(three_months_ago)
    idx_3m = min(max(int(idx_3m), 0), len(idx) - 1)
    price_3m_ago = close.iloc[idx_3m]
    return_3m = (price_today / price_3m_ago - 1) * 100 if price_3m_ago else None

    rsi_30d_series = rsi(close, period=30)
    rsi_14d_series = rsi(close, period=14)
    if len(rsi_14d_series) < 8 or pd.isna(rsi_30d_series.iloc[-1]) or pd.isna(rsi_14d_series.iloc[-1]):
        raise ValueError("insufficient data to compute RSI")

    rsi_30d_val = float(rsi_30d_series.iloc[-1])
    rsi_14d_val = float(rsi_14d_series.iloc[-1])
    # "3 days ago" / "7 days ago" as trading-day offsets, consistent with the
    # rest of this module operating on daily bars rather than calendar days.
    rsi_14d_3d_ago = rsi_14d_series.iloc[-4]
    rsi_14d_7d_ago = rsi_14d_series.iloc[-8]

    if rsi_14d_val > rsi_14d_3d_ago > rsi_14d_7d_ago:
        direction = "rising"
    elif rsi_14d_val < rsi_14d_3d_ago < rsi_14d_7d_ago:
        direction = "falling"
    else:
        direction = "neutral"

    avg_volume_30d = volume.tail(30).mean()
    if pd.isna(avg_volume_30d):
        raise ValueError("insufficient volume history")
    avg_volume_30d = float(avg_volume_30d)

    passes = (
        settings.DISLOCATION_ZONE_MAX <= return_1y <= settings.DISLOCATION_ZONE_MIN
        and rsi_30d_val <= rsi_threshold
        and avg_volume_30d >= settings.UNIVERSE_MIN_AVG_VOLUME
    )

    price_action_type = _classify_price_action(return_1y, return_3m) if return_3m is not None else None

    # Multi-factor entry timing — computed here (not in screener.py) while
    # the full RSI/volume/price series are still in scope; only the scalar
    # timing_signal/timing_note survive into the cached CSV row.
    timing = classify_timing(
        rsi_14d_val, direction,
        rsi_14d_series=rsi_14d_series, volume_series=volume, price_series=close,
    )

    return {
        "ticker": ticker,
        "return_1y": return_1y,
        "return_3m": return_3m,
        "price_action_type": price_action_type,
        "rsi_30d": rsi_30d_val,
        "rsi_14d": rsi_14d_val,
        "rsi_14d_direction": direction,
        "avg_volume_30d": avg_volume_30d,
        "passes_dislocation": passes,
        "timing_signal": timing["timing_signal"],
        "timing_note": timing["timing_note"],
    }


def _fetch_market_cap(ticker: str) -> tuple[float, bool]:
    """Return (market_cap_b, unverified) for one ticker via yfinance `.info`.

    `.info` is a single-ticker network call with no batch equivalent, so
    this is only called on names that already passed the price/RSI/volume
    filter, not the full universe. A request failure or a missing/zero
    `marketCap` marks the row `unverified` rather than excluding it — we
    don't have evidence the company is actually too small, only that
    yfinance didn't give us a market cap for it.
    """
    try:
        info = yf.Ticker(ticker).info
    except Exception as exc:  # yfinance can raise a variety of things on network/format issues
        logger.warning("Market cap fetch failed for %s: %s — leaving unverified", ticker, exc)
        return 0.0, True

    market_cap = info.get("marketCap") or 0
    if not market_cap:
        logger.warning("No marketCap from yfinance for %s — leaving unverified", ticker)
        return 0.0, True

    return market_cap / 1e9, False


def _process_batch(batch: list[str], rsi_threshold: float) -> list[dict]:
    try:
        data = yf_download(
            batch,
            period=YF_DOWNLOAD_PERIOD,
            group_by="ticker",
            auto_adjust=True,
            threads=True,
            progress=False,
        )
    except Exception as exc:  # yfinance can raise a variety of things on network/format issues
        logger.warning("yf.download failed for batch of %d tickers: %s", len(batch), exc)
        return []

    if data.empty:
        logger.warning("yf.download returned no data for batch starting with %s", batch[0])
        return []

    rows = []
    for ticker in batch:
        try:
            sub = _extract_ticker_frame(data, ticker, single_ticker=len(batch) == 1)
            rows.append(_compute_row(ticker, sub, rsi_threshold))
        except Exception as exc:
            logger.warning("Skipping %s: %s", ticker, exc)
            continue
    return rows


def _run_price_filter_uncached(tickers: list[str], batch_size: int, rsi_threshold: float) -> pd.DataFrame:
    rows: list[dict] = []
    for batch in _chunk(tickers, batch_size):
        rows.extend(_process_batch(batch, rsi_threshold))

    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = pd.DataFrame(rows)
    passing = df[df["passes_dislocation"]].reset_index(drop=True)

    # Market cap filter — real company market cap from yfinance, fetched
    # only for the (small) set of names that already passed price/RSI/volume.
    # See module docstring for why this can't happen in universe.py.
    market_caps = [_fetch_market_cap(t) for t in passing["ticker"]]
    passing["market_cap_b"] = [mc for mc, _ in market_caps]
    passing["market_cap_unverified"] = [unverified for _, unverified in market_caps]

    keep = passing["market_cap_unverified"] | (passing["market_cap_b"] >= settings.UNIVERSE_MIN_MARKET_CAP_B)
    passing = passing[keep].reset_index(drop=True)

    return passing[OUTPUT_COLUMNS]


def run_price_filter(
    tickers: list[str], batch_size: int = 100, regime_flags: list[str] | None = None
) -> pd.DataFrame:
    """Run the price/RSI/volume dislocation pre-filter over `tickers`.

    Fetches price history from yfinance in batches of `batch_size` via
    `yf.download()`. Returns only the tickers that PASS the dislocation
    filter (see module docstring) AND clear the `UNIVERSE_MIN_MARKET_CAP_B`
    market cap floor (or have `market_cap_unverified=True`, in which case
    they pass through regardless — see `_fetch_market_cap`). Columns: see
    `OUTPUT_COLUMNS`.

    `regime_flags` (today's active flags from `market_snapshot.fetch_market_snapshot()`)
    adjusts the RSI 30D dislocation threshold via
    `_get_rsi_threshold()`/`market_config.RSI_30D_THRESHOLD_BY_REGIME` —
    a more volatile regime demands a deeper oversold reading to pass.
    Falls back to `settings.RSI_30D_MAX` when `regime_flags` is None/empty.

    Individual ticker failures (no data, insufficient history, etc.) are
    logged and skipped — never crash the whole run over one bad ticker.

    Cached to `equity/data/cache/price_filter_{date}.csv` for
    `settings.PRICE_CACHE_HOURS` hour(s). A run/pass/pass-rate summary for
    this call is attached to `result.attrs["summary"]` and logged.
    """
    rsi_threshold = _get_rsi_threshold(regime_flags)

    cache_path = _cache_path()
    if _is_cache_fresh(cache_path):
        logger.info("Loading price filter results from fresh cache: %s", cache_path)
        result = pd.read_csv(cache_path)
    else:
        result = _run_price_filter_uncached(tickers, batch_size, rsi_threshold)
        result.to_csv(cache_path, index=False)
        logger.info("Cached price filter results to %s", cache_path)

    total_in = len(tickers)
    passed = len(result)
    pass_rate_pct = (passed / total_in * 100) if total_in else 0.0
    result.attrs["summary"] = {"total_in": total_in, "passed": passed, "pass_rate_pct": pass_rate_pct}
    logger.info("Price filter: %d/%d passed (%.1f%%)", passed, total_in, pass_rate_pct)
    return result


if __name__ == "__main__":
    from equity.screener.universe import get_universe_tickers

    logging.basicConfig(level=logging.INFO)

    start = time.time()
    universe_tickers = get_universe_tickers()
    filtered = run_price_filter(universe_tickers)
    elapsed = time.time() - start

    total_in = len(universe_tickers)
    passed = len(filtered)
    pass_rate_pct = (passed / total_in * 100) if total_in else 0.0

    print(f"Total tickers: {total_in}")
    print(f"Passed: {passed}")
    print(f"Pass rate: {pass_rate_pct:.1f}%")

    print("\nTop 10 passing names by most negative 1Y return:")
    top10 = filtered.sort_values("return_1y").head(10)
    print(top10.to_string(index=False))

    print(f"\nTime taken: {elapsed:.1f}s")
