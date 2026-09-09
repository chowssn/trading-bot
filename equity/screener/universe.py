"""Russell 1000 universe from iShares' IWB holdings CSV.

Fetches the iShares Russell 1000 ETF (IWB) full holdings list, which stands
in for the Russell 1000 index membership (IWB is a close-to-exact physical
replication of the index, and Russell doesn't offer a free constituent
feed). This is the first, cheapest stage of the equity screener funnel —
everything downstream (`price_filter.py`, and later the FMP fundamental
screen) narrows from this list.

The iShares CSV has a distinctive shape: a handful of metadata rows (fund
name, inception date, etc.) precede the actual holdings table, and the
holdings table is followed by disclaimer/footer text or futures/cash line
items whose ticker field isn't a real ticker (blank, '-', or numeric like
'FAU6'). We locate the real header row by scanning for the row whose first
column reads 'Ticker', and stop consuming rows at the first ticker after
that which isn't 1-5 uppercase letters. Non-equity rows *within* the table
(e.g. a cash sweep line with a normal-looking ticker) aren't caught by this
and are instead dropped by the `asset_class == 'Equity'` filter.

Note on `market_value`: this is IWB's dollar position in that name (share
count x IWB's holding), not the company's total market cap — iShares
doesn't publish market cap directly, and this column is NOT usable as a
market cap proxy (IWB's largest position is ~$3B against ~$49B total AUM,
so even a modest true-market-cap floor would reject every row). It's kept
in the output for reference only. Market cap filtering happens downstream
in `price_filter.py`, using real company market cap from yfinance, applied
only to the small set of names that already pass the price/RSI/volume
dislocation screen.
"""

import logging
import re
import time
from pathlib import Path

import pandas as pd
import requests

from equity.config import settings

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "universe"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CACHE_FILE = DATA_DIR / "iwb_holdings.csv"
CACHE_TTL_SECONDS = settings.UNIVERSE_CACHE_HOURS * 60 * 60

REQUEST_TIMEOUT_SECONDS = 30
# iShares' CSV endpoint 403s on the default python-requests UA.
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

# Raw iShares column name -> our column name.
COLUMN_MAP = {
    "ticker": "ticker",
    "name": "name",
    "sector": "sector",
    "asset class": "asset_class",
    "weight (%)": "weight_pct",
    "market value": "market_value",
}

OUTPUT_COLUMNS = ["ticker", "name", "sector", "asset_class", "weight_pct", "market_value"]

# Tickers confirmed unavailable on yfinance — either genuinely delisted
# or taken private. Remove from this set if/when they return to public markets.
# Last verified: 2026-08-30
DELISTED_TICKERS = {
    'HOLX',   # Hologic — taken private by Blackstone, deal closed late 2025
}

# A tradable US-equity ticker as it appears in this feed: 1-5 uppercase
# letters, optionally followed by a single-letter share-class suffix
# separated by a space, dot, or hyphen ("BRK B", "HEI A", "BF B"). Used to
# find where the real holdings table ends (see `_parse_raw_holdings`) —
# futures/cash line items sorted to the tail (e.g. "FAU6", "ESU6") fail
# this on their digits, while non-equity rows shuffled into the middle of
# the table (e.g. "USD", "XTSLA") pass it and are dropped later by the
# asset_class == 'Equity' filter instead.
#
# The class-suffix branch is NOT cosmetic: iShares switched multi-class
# tickers from the concatenated form ("BRKB") to a space-separated one
# ("BRK B") in the feed. Because `_parse_raw_holdings` truncates the table
# at the FIRST ticker that fails this pattern, and Berkshire sits at row
# ~12 by weight, the old letters-only pattern silently cut the universe
# down to 11 names — the screener's "scanning 11 names" symptom. Any future
# narrowing of this pattern risks the same silent truncation; the
# `MIN_EXPECTED_TICKERS` guard in `fetch_russell_1000()` is the backstop.
TICKER_PATTERN = re.compile(r"^[A-Z]{1,5}(?:[ .\-][A-Z])?$")

# A parsed universe smaller than this means the feed shape changed (or the
# endpoint returned a stub) rather than the index genuinely shrinking —
# the Russell 1000 has ~1000 names. Below it we refuse to trust the fetch,
# and in particular refuse to overwrite a good cache with it.
MIN_EXPECTED_TICKERS = 100

# Share-class separators iShares uses that yfinance expects as a hyphen.
_CLASS_SUFFIX_RE = re.compile(r"^([A-Z]{1,5})[ .]([A-Z])$")

# Row-level tickers that show up in the holdings table but are obviously not
# tradable US equities (cash sweep, futures margin, etc.), belt-and-suspenders
# on top of the asset_class == 'Equity' filter.
NON_EQUITY_TICKER_BLACKLIST = {"USD", "CASH"}

# Maps iShares CSV ticker -> yfinance ticker format.
# iShares sometimes omits hyphens/dots that yfinance requires for
# multi-class share tickers. Applied in _clean_holdings() AFTER the
# TICKER_PATTERN/valid_ticker filters above run on the original iShares
# format (those filters would reject 'BRK-B' or 'HEI.A' outright), so the
# untranslated key is what needs to look like a plain 1-5 letter ticker.
# Add new entries here when new mismatches are discovered.
TICKER_TRANSLATION = {
    'BRKB':  'BRK-B',   # Berkshire Hathaway Class B
    'BRKA':  'BRK-A',   # Berkshire Hathaway Class A
    'HEIA':  'HEI-A',   # yfinance uses HEI-A not HEI.A
    'BFB':   'BF-B',    # Brown-Forman Class B
    'BFA':   'BF-A',    # Brown-Forman Class A
    'LENB':  'LEN-B',   # Lennar Corp Class B
    'UHALB': 'UHAL-B',  # U-Haul Holding Class B
}
# HOLX (Hologic) intentionally has no entry here: the iShares ticker was
# already in yfinance's expected format (plain 'HOLX'), so this was never a
# ticker-format mismatch. HOLX went private (Blackstone, late 2025) and is
# now excluded upstream via DELISTED_TICKERS instead.


def _normalize_ticker(ticker: str) -> str:
    """iShares ticker -> yfinance ticker.

    Two steps, in order:

    1. An explicit `TICKER_TRANSLATION` entry wins if one exists (that map
       still carries the concatenated forms — 'BRKB', 'HEIA' — the feed used
       before it switched to space separators, so a revert doesn't re-break
       anything).
    2. Otherwise a space/dot share-class separator becomes a hyphen, which
       is the form yfinance wants: 'BRK B' -> 'BRK-B', 'HEI A' -> 'HEI-A'.

    Anything else passes through unchanged.
    """
    if ticker in TICKER_TRANSLATION:
        return TICKER_TRANSLATION[ticker]
    match = _CLASS_SUFFIX_RE.match(ticker)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    return ticker


def _is_cache_fresh(path: Path) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < CACHE_TTL_SECONDS


def _download_universe_csv() -> str:
    resp = requests.get(settings.universe_url(), headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT_SECONDS)
    resp.raise_for_status()
    return resp.text


def _find_header_row(lines: list[str]) -> int:
    """Index of the row whose first CSV field is 'Ticker'."""
    for i, line in enumerate(lines):
        first_field = line.split(",")[0].strip().strip('"')
        if first_field == "Ticker":
            return i
    raise RuntimeError("Could not locate 'Ticker' header row in iShares holdings CSV")


def _parse_raw_holdings(csv_text: str) -> pd.DataFrame:
    """Parse the iShares CSV text into a raw holdings DataFrame.

    Reads from the 'Ticker' header row through EOF, skipping any malformed
    rows (the footer disclaimer paragraph has a different shape than the
    data rows and gets dropped here automatically), then truncates at the
    first row whose ticker isn't 1-5 uppercase letters — futures/cash line
    items sorted to the tail (e.g. 'FAU6') and any remaining footer content
    both look like this and mark the end of real holdings.
    """
    lines = csv_text.splitlines()
    header_idx = _find_header_row(lines)

    from io import StringIO

    df = pd.read_csv(
        StringIO(csv_text),
        skiprows=header_idx,
        engine="python",
        on_bad_lines="skip",
        thousands=",",
    )
    df.columns = [str(c).strip() for c in df.columns]

    if "Ticker" not in df.columns:
        raise RuntimeError("Parsed iShares CSV is missing a 'Ticker' column after header detection")

    ticker_col = df["Ticker"].astype(str).str.strip().str.strip('"').str.upper()
    valid_mask = ticker_col.str.match(TICKER_PATTERN)
    rows_before_truncation = len(df)
    if not valid_mask.all():
        first_invalid = (~valid_mask).idxmax()
        # Log WHAT stopped the table and WHERE. A truncation this early is
        # the failure mode that quietly shrank the universe to 11 names
        # (see TICKER_PATTERN) — surfacing the offending ticker makes the
        # next feed-format change a one-line diagnosis instead of a hunt.
        logger.info(
            "Universe parse: table truncated at row %s of %d on ticker %r",
            first_invalid, rows_before_truncation, ticker_col.loc[first_invalid],
        )
        df = df.loc[: first_invalid - 1] if first_invalid > df.index[0] else df.iloc[0:0]

    logger.info(
        "Universe parse: %d raw rows from CSV, %d rows kept after end-of-table truncation",
        rows_before_truncation, len(df),
    )
    return df


def _clean_holdings(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Rename to our schema, coerce types, and apply universe filters."""
    normalized = {str(c).strip().lower(): c for c in df_raw.columns}

    missing = [k for k in COLUMN_MAP if k not in normalized]
    if missing:
        raise RuntimeError(f"iShares CSV is missing expected column(s): {missing} (have: {list(df_raw.columns)})")

    df = df_raw.rename(columns={normalized[k]: v for k, v in COLUMN_MAP.items()})[OUTPUT_COLUMNS].copy()

    df["ticker"] = df["ticker"].astype(str).str.strip().str.strip('"').str.upper()
    df["name"] = df["name"].astype(str).str.strip()
    df["sector"] = df["sector"].astype(str).str.strip()
    df["asset_class"] = df["asset_class"].astype(str).str.strip()

    for col in ("weight_pct", "market_value"):
        df[col] = (
            df[col].astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    rows_in = len(df)

    # Asset class: equities only.
    df = df[df["asset_class"] == "Equity"]
    after_equity = len(df)

    # Translate iShares tickers to yfinance format. Runs BEFORE the
    # validity filter below, not after: the feed's share-class tickers
    # arrive as "BRK B"/"HEI A", and the validity filter rejects residual
    # spaces and dots — so normalizing afterwards would mean those names
    # were already dropped. See `_normalize_ticker()`.
    df["ticker"] = df["ticker"].map(_normalize_ticker)

    # Remove confirmed delisted/private tickers (matched on the normalized
    # yfinance form, which is what DELISTED_TICKERS holds).
    df = df[~df["ticker"].isin(DELISTED_TICKERS)]

    # Ticker validity: non-blank, not a known non-equity symbol, and no
    # residual space/dot — a hyphen is fine (it's the normalized share-class
    # form), but anything still carrying a separator after normalization is
    # a shape this parser doesn't understand and shouldn't guess at.
    valid_ticker = (
        df["ticker"].notna()
        & ~df["ticker"].isin(["", "-", "NAN", "NONE"])
        & ~df["ticker"].str.contains(r"\.", regex=True)
        & ~df["ticker"].str.contains(" ", regex=False)
        & ~df["ticker"].isin(NON_EQUITY_TICKER_BLACKLIST)
    )
    dropped = df.loc[~valid_ticker, "ticker"].tolist()
    if dropped:
        logger.debug("Universe: dropped %d row(s) on ticker validity: %s", len(dropped), dropped[:20])
    df = df[valid_ticker]
    after_valid = len(df)

    # No market cap filter here — see module docstring: market_value is
    # IWB's position size, not company market cap. Market cap filtering
    # happens downstream in price_filter.py using yfinance data.
    df = df.drop_duplicates(subset="ticker").reset_index(drop=True)

    logger.info(
        "Universe clean: %d parsed rows -> %d equity -> %d valid ticker -> %d after dedup",
        rows_in, after_equity, after_valid, len(df),
    )
    return df


def fetch_russell_1000(force_refresh: bool = False) -> pd.DataFrame:
    """Return the cleaned Russell 1000 (IWB) equity universe.

    Downloads and parses the iShares IWB holdings CSV, applying universe
    filters (equity-only, valid US ticker format). No market cap filter is
    applied here — market_value is IWB's position size, not company market
    cap (see module docstring); market cap filtering happens downstream in
    `price_filter.py` using real market cap from yfinance. Cached to
    `equity/data/universe/iwb_holdings.csv` for `settings.UNIVERSE_CACHE_HOURS`
    hours.

    If the download fails and a cache exists (even stale), falls back to it
    with a warning. If the download fails and there is no cache, raises
    RuntimeError.
    """
    if not force_refresh and _is_cache_fresh(CACHE_FILE):
        logger.info("Loading Russell 1000 universe from fresh cache: %s", CACHE_FILE)
        return pd.read_csv(CACHE_FILE)

    try:
        csv_text = _download_universe_csv()
        df_raw = _parse_raw_holdings(csv_text)
        df = _clean_holdings(df_raw)
    except (requests.RequestException, RuntimeError) as exc:
        if CACHE_FILE.exists():
            logger.warning(
                "Failed to refresh Russell 1000 universe (%s) — falling back to stale cache: %s", exc, CACHE_FILE
            )
            return pd.read_csv(CACHE_FILE)
        raise RuntimeError(f"Failed to fetch Russell 1000 universe and no cache exists at {CACHE_FILE}: {exc}") from exc

    # Sanity gate. The download can succeed (HTTP 200, parses cleanly) and
    # still yield a near-empty universe if the feed's shape changed — that
    # is what happened when iShares switched to space-separated share-class
    # tickers and the parser truncated the table at row 12. A short result
    # is therefore treated as a failed fetch, not as the index shrinking:
    # we fall back to cache and, critically, do NOT overwrite the cache with
    # it (otherwise one bad fetch poisons every later run's fallback too).
    if len(df) < MIN_EXPECTED_TICKERS:
        logger.error(
            "Universe fetch returned only %d tickers (expected >= %d) — likely a feed-format "
            "change or an endpoint stub. Not caching this result.",
            len(df), MIN_EXPECTED_TICKERS,
        )
        if CACHE_FILE.exists():
            cached = pd.read_csv(CACHE_FILE)
            if len(cached) >= MIN_EXPECTED_TICKERS:
                logger.warning("Falling back to cached universe: %d tickers from %s", len(cached), CACHE_FILE)
                return cached
            logger.error(
                "Cached universe at %s is also short (%d tickers) — no good fallback available.",
                CACHE_FILE, len(cached),
            )
        raise RuntimeError(
            f"Russell 1000 universe fetch returned only {len(df)} tickers "
            f"(expected >= {MIN_EXPECTED_TICKERS}) and no usable cache exists at {CACHE_FILE}"
        )

    df.to_csv(CACHE_FILE, index=False)
    logger.info("Universe fetch: %d tickers returned, cached to %s", len(df), CACHE_FILE)
    return df


def get_universe_tickers() -> list[str]:
    """Just the ticker list from `fetch_russell_1000()`."""
    return fetch_russell_1000()["ticker"].tolist()


def translate_ticker(ticker: str) -> str:
    """Translates an iShares CSV ticker to yfinance format. Returns unchanged if no translation applies."""
    return _normalize_ticker(ticker)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    universe = fetch_russell_1000()
    print(f"Shape: {universe.shape}")
    print(universe.head(5))
    print(f"Total tickers: {len(universe)}")
