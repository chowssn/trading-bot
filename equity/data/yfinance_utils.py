"""yfinance wrapper with retry logic for transient failures and rate limits.

yfinance occasionally returns an empty DataFrame or raises on a rate limit
(HTTP 429/503) or a transient network blip rather than a clean error, so a
bare `yf.download()`/`.history()` call silently degrades to "no data" one
retry away from succeeding. `yf_download()` and `yf_ticker_history()` are
drop-in replacements that retry with exponential backoff before giving up.

Only the batch download calls in the brief pipeline and screener — the
highest-volume, most rate-limit-prone callers — have been switched over so
far (`equity/screener/price_filter.py`, `equity/brief/performance_tracker.py`,
`equity/brief/market_snapshot.py`). Single-ticker `.history()` calls
elsewhere (`advisor.py`, `monitor.py`) can move to `yf_ticker_history()`
incrementally.
"""

import logging
import time

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def yf_download(tickers: list[str] | str,
                 period: str = "1y",
                 interval: str = "1d",
                 auto_adjust: bool = True,
                 progress: bool = False,
                 max_retries: int = 3,
                 retry_delay: float = 5.0,
                 **kwargs) -> pd.DataFrame:
    """
    Drop-in replacement for `yf.download()` with exponential backoff retry.

    Retries on both an empty result (yfinance sometimes returns an empty
    frame on rate limit rather than raising) and a raised exception
    (network error, timeout, HTTP 429/503). Returns an empty DataFrame —
    never raises — once `max_retries` is exhausted, matching how callers
    already treat a `yf.download()` empty-result response.

    Args:
        max_retries: number of retry attempts after the first try (default 3).
        retry_delay: base delay in seconds, doubles each retry (default 5s,
            i.e. 5s/10s/20s).
        All other args are passed directly through to `yf.download()`.
    """
    last_error = None

    for attempt in range(max_retries + 1):
        if attempt > 0:
            delay = retry_delay * (2 ** (attempt - 1))
            logger.warning(
                "yf_download: attempt %d/%d after %.0fs delay (tickers: %s)",
                attempt + 1, max_retries + 1, delay,
                tickers if isinstance(tickers, str) else len(tickers),
            )
            time.sleep(delay)

        try:
            result = yf.download(
                tickers,
                period=period,
                interval=interval,
                auto_adjust=auto_adjust,
                progress=progress,
                **kwargs,
            )
        except Exception as exc:  # yfinance can raise a variety of things on network/format issues
            last_error = str(exc)
            logger.warning("yf_download: exception on attempt %d: %s", attempt + 1, exc)
            continue

        if result is None or result.empty:
            last_error = "empty DataFrame returned"
            logger.warning("yf_download: empty result on attempt %d", attempt + 1)
            continue

        if attempt > 0:
            logger.info("yf_download: succeeded on attempt %d", attempt + 1)
        return result

    logger.error(
        "yf_download: all %d attempts failed. Last error: %s. Tickers: %s",
        max_retries + 1, last_error,
        tickers if isinstance(tickers, str) else tickers[:5],
    )
    return pd.DataFrame()


def yf_ticker_history(ticker: str,
                       period: str = "1y",
                       interval: str = "1d",
                       auto_adjust: bool = True,
                       max_retries: int = 3,
                       retry_delay: float = 3.0) -> pd.DataFrame:
    """
    Wrapper around `yf.Ticker(ticker).history()` with retry logic.
    Use for single-ticker history fetches.
    """
    last_error = None

    for attempt in range(max_retries + 1):
        if attempt > 0:
            delay = retry_delay * (2 ** (attempt - 1))
            time.sleep(delay)

        try:
            result = yf.Ticker(ticker).history(
                period=period,
                interval=interval,
                auto_adjust=auto_adjust,
            )
            if result is not None and not result.empty:
                return result
            last_error = "empty history returned"
        except Exception as exc:  # yfinance can raise a variety of things on network/format issues
            last_error = str(exc)

    logger.warning(
        "yf_ticker_history(%s): failed after %d attempts: %s",
        ticker, max_retries + 1, last_error,
    )
    return pd.DataFrame()
