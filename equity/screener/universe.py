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
# letters. Used to find where the real holdings table ends (see
# `_parse_raw_holdings`) — futures/cash line items sorted to the tail
# (e.g. "FAU6", "ESU6") fail this, while non-equity rows shuffled into the
# middle of the table (e.g. "USD", "XTSLA") pass it and are dropped later
# by the asset_class == 'Equity' filter instead.
TICKER_PATTERN = re.compile(r"^[A-Z]{1,5}$")

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
    if not valid_mask.all():
        first_invalid = (~valid_mask).idxmax()
        df = df.loc[: first_invalid - 1] if first_invalid > df.index[0] else df.iloc[0:0]

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

    # Asset class: equities only.
    df = df[df["asset_class"] == "Equity"]

    # Remove confirmed delisted/private tickers
    df = df[~df['ticker'].isin(DELISTED_TICKERS)]

    # Ticker validity: non-blank, no dots, no spaces, not a known non-equity symbol.
    valid_ticker = (
        df["ticker"].notna()
        & ~df["ticker"].isin(["", "-", "NAN", "NONE"])
        & ~df["ticker"].str.contains(r"\.", regex=True)
        & ~df["ticker"].str.contains(" ", regex=False)
        & ~df["ticker"].isin(NON_EQUITY_TICKER_BLACKLIST)
    )
    df = df[valid_ticker]

    # No market cap filter here — see module docstring: market_value is
    # IWB's position size, not company market cap. Market cap filtering
    # happens downstream in price_filter.py using yfinance data.
    df = df.drop_duplicates(subset="ticker").reset_index(drop=True)

    # Translate iShares tickers to yfinance format. Runs last, after the
    # ticker-validity filters and dedup above (which operate on the
    # original iShares format) — see TICKER_TRANSLATION comment.
    df["ticker"] = df["ticker"].map(lambda t: TICKER_TRANSLATION.get(t, t))

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

    df.to_csv(CACHE_FILE, index=False)
    logger.info("Fetched Russell 1000 universe: %d tickers, cached to %s", len(df), CACHE_FILE)
    return df


def get_universe_tickers() -> list[str]:
    """Just the ticker list from `fetch_russell_1000()`."""
    return fetch_russell_1000()["ticker"].tolist()


def translate_ticker(ticker: str) -> str:
    """Translates an iShares CSV ticker to yfinance format. Returns unchanged if no translation exists."""
    return TICKER_TRANSLATION.get(ticker, ticker)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    universe = fetch_russell_1000()
    print(f"Shape: {universe.shape}")
    print(universe.head(5))
    print(f"Total tickers: {len(universe)}")
