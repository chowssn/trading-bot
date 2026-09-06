"""One-off classifier: fill in blank `sector` fields in positions_override.json.

IBKR-imported positions (see `ibkr_positions_import.py`) land with
`"sector": ""` because IBKR's export doesn't carry GICS classification.
This script backfills that field from yfinance's `Ticker.info['sector']`,
mapped onto our own sector taxonomy (`YFINANCE_SECTOR_MAP`), with a
hardcoded fallback (`ETF_SECTORS`) for tickers yfinance has no sector for
— ETFs, plus a couple of ADRs/OTC names with sparse yfinance data.

A ticker is only touched if it's genuinely unclassified: blank/missing in
positions_override.json *and* blank in the merged view (`equity.config.
positions`, which layers the override over positions.py's hand-curated
entries). Tickers like CCJ/CEG/MSFT/PGR/UMAC/APP already carry a real,
thesis-driven sector in positions.py itself and are left alone — this
script must never clobber a curated classification with a generic
yfinance one.

Also updates `market_config.py`'s `POSITION_SECTOR_MAP` (ticker -> sector
benchmark ETF) for any newly classified, non-ETF ticker whose sector has a
known SPDR-style benchmark (`SECTOR_ETF_MAP`) and isn't already in the map.
This edits market_config.py's source directly rather than going through
`config_manager.update_market_config()`, because that function can only
replace the value at an *existing* path — it has no support for inserting
new dict entries. The insertion here uses the same approach in spirit
(locate the exact `ast` span, splice, then re-parse to validate before
writing) but isn't wired into config_manager's changelog/git-commit flow,
since this is a manual, run-once tool, not an AI-assisted/approved config
change. Review the diff and commit it yourself.

Run manually after an IBKR positions import surfaces new tickers. Not part
of the daily pipeline.
"""

from __future__ import annotations

import ast
import json
import time
from pathlib import Path

import yfinance as yf

from equity.config import market_config, positions

REPO_ROOT = Path(__file__).resolve().parents[2]
OVERRIDE_PATH = REPO_ROOT / "equity" / "config" / "positions_override.json"
MARKET_CONFIG_PATH = REPO_ROOT / "equity" / "config" / "market_config.py"

YFINANCE_SECTOR_KEY = "sector"

# yfinance `info['sector']` -> our standard sector taxonomy (matches the
# `sector` values hand-written in positions.py, e.g. "Information
# Technology" for MSFT, "Financials" for PGR).
YFINANCE_SECTOR_MAP = {
    "Technology": "Information Technology",
    "Communication Services": "Communication Services",
    "Consumer Cyclical": "Consumer Cyclical",
    "Consumer Defensive": "Consumer Staples",
    "Financial Services": "Financials",
    "Healthcare": "Healthcare",
    "Industrials": "Industrials",
    "Basic Materials": "Basic Materials",
    "Energy": "Energy",
    "Real Estate": "Real Estate",
    "Utilities": "Utilities",
    "ETF": "ETF",
}

# Hardcoded fallback for tickers yfinance returns no usable sector for:
# ETFs (QQQ, SPY, TLT, SMH, EWW, URA, PPLT, SKHY) have no GICS sector at
# all, and BYDDY (BYD ADR, OTC) has spotty yfinance coverage. These
# short-circuit the yfinance lookup entirely — no API call is made for
# them.
ETF_SECTORS = {
    "QQQ": "Technology",
    "SPY": "Broad Market",
    "TLT": "Government Bond",
    "SMH": "Semiconductors",
    "EWW": "Emerging Markets",
    "URA": "Uranium/Nuclear",
    "PPLT": "Precious Metals",
    "SKHY": "Technology",
    "BYDDY": "Consumer Cyclical",
}

# Subset of ETF_SECTORS that are actual ETF holdings (as opposed to
# BYDDY, a regular equity). These are excluded from the POSITION_SECTOR_MAP
# update below — an ETF position doesn't need a sector-ETF benchmark to
# compare itself against; it effectively *is* one.
_ACTUAL_ETFS = {"QQQ", "SPY", "TLT", "SMH", "EWW", "URA", "PPLT", "SKHY"}

# Standard sector name -> (benchmark ETF ticker, POSITION_SECTOR_MAP
# comment), for newly classified non-ETF tickers. Mirrors market_config's
# own BENCHMARK_TICKERS sector ETFs. No entry for "Consumer Staples" —
# market_config.BENCHMARK_TICKERS doesn't carry XLP.
SECTOR_ETF_MAP = {
    "Information Technology": ("XLK", "tech ETF"),
    "Communication Services": ("XLC", "communication services ETF"),
    "Consumer Cyclical": ("XLY", "consumer discretionary ETF"),
    "Financials": ("XLF", "financials ETF"),
    "Healthcare": ("XLV", "healthcare ETF"),
    "Industrials": ("XLI", "industrials ETF"),
    "Basic Materials": ("XLB", "materials ETF"),
    "Energy": ("XLE", "energy ETF"),
    "Real Estate": ("XLRE", "real estate ETF"),
    "Utilities": ("XLU", "utilities ETF"),
    "Uranium/Nuclear": ("URA", "uranium ETF"),
}

YFINANCE_REQUEST_DELAY_SEC = 0.3  # be polite to the yfinance/Yahoo endpoint


def _load_override() -> dict:
    if not OVERRIDE_PATH.exists():
        return {"POSITIONS": {}, "WATCHLIST": {}}
    with open(OVERRIDE_PATH) as f:
        data = json.load(f)
    data.setdefault("POSITIONS", {})
    data.setdefault("WATCHLIST", {})
    return data


def _save_override(data: dict) -> None:
    with open(OVERRIDE_PATH, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def _needs_classification(ticker: str, override_entry: dict) -> bool:
    """True if `ticker` has no sector, either in the override file or merged."""
    if override_entry.get("sector"):
        return False
    merged = positions.get_position(ticker) or {}
    return not merged.get("sector")


def _fetch_yfinance_sector(ticker: str) -> str:
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        return ""
    return info.get(YFINANCE_SECTOR_KEY, "") or ""


def classify_sector(ticker: str) -> tuple[str, str]:
    """Return (sector, note). `sector` is '' if the ticker couldn't be mapped."""
    if ticker in ETF_SECTORS:
        return ETF_SECTORS[ticker], "hardcoded"

    raw = _fetch_yfinance_sector(ticker)
    if not raw:
        return "", "yfinance returned no sector"

    mapped = YFINANCE_SECTOR_MAP.get(raw)
    if not mapped:
        return "", f"yfinance sector {raw!r} has no entry in YFINANCE_SECTOR_MAP"

    return mapped, "yfinance"


def _insert_position_sector_map_entries(source: str, new_entries: dict[str, tuple[str, str]]) -> str | None:
    """Splice `new_entries` ({ticker: (etf, comment)}) into POSITION_SECTOR_MAP.

    Returns the edited source, or None if POSITION_SECTOR_MAP isn't found or
    the edit would produce invalid Python (nothing is written either way).
    """
    if not new_entries:
        return source

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    target = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Dict)
            and any(isinstance(t, ast.Name) and t.id == "POSITION_SECTOR_MAP" for t in node.targets)
        ):
            target = node.value
            break

    if target is None:
        return None

    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))

    # Insert right before the closing-brace line.
    insert_at = offsets[target.end_lineno - 1]
    inserted = "".join(
        f"    {ticker!r}: {etf!r},    # {comment}\n" for ticker, (etf, comment) in new_entries.items()
    )
    new_source = source[:insert_at] + inserted + source[insert_at:]

    try:
        ast.parse(new_source)
    except SyntaxError:
        return None

    return new_source


def update_market_config(new_sectors: dict[str, str]) -> dict[str, str]:
    """Add POSITION_SECTOR_MAP entries for newly classified, non-ETF tickers.

    `new_sectors` is {ticker: sector} for tickers just classified this run.
    Returns {ticker: etf_ticker} for entries actually added.
    """
    existing_map = market_config.POSITION_SECTOR_MAP

    candidates = {}
    for ticker, sector in new_sectors.items():
        if ticker in _ACTUAL_ETFS or ticker in existing_map:
            continue
        etf_comment = SECTOR_ETF_MAP.get(sector)
        if etf_comment is None:
            continue
        candidates[ticker] = etf_comment

    if not candidates:
        return {}

    source = MARKET_CONFIG_PATH.read_text()
    new_source = _insert_position_sector_map_entries(source, candidates)
    if new_source is None:
        return {}

    MARKET_CONFIG_PATH.write_text(new_source)
    return {ticker: etf for ticker, (etf, _comment) in candidates.items()}


def main() -> None:
    override = _load_override()

    to_classify = [
        ticker for ticker, entry in override["POSITIONS"].items() if _needs_classification(ticker, entry)
    ]

    updated: dict[str, str] = {}
    unmapped: dict[str, str] = {}

    for ticker in to_classify:
        sector, note = classify_sector(ticker)
        if sector:
            override["POSITIONS"][ticker]["sector"] = sector
            updated[ticker] = sector
        else:
            unmapped[ticker] = note

        if ticker not in ETF_SECTORS:
            time.sleep(YFINANCE_REQUEST_DELAY_SEC)

    if updated:
        _save_override(override)

    added_to_market_config = update_market_config(updated)

    print(f"Checked {len(to_classify)} unclassified ticker(s) in positions_override.json")
    print()
    print(f"Updated ({len(updated)}):")
    for ticker, sector in updated.items():
        print(f"  {ticker}: {sector}")
    if not updated:
        print("  (none)")
    print()
    print(f"Couldn't be mapped ({len(unmapped)}):")
    for ticker, note in unmapped.items():
        print(f"  {ticker}: {note}")
    if not unmapped:
        print("  (none)")
    print()
    print(f"market_config.py POSITION_SECTOR_MAP additions ({len(added_to_market_config)}):")
    for ticker, etf in added_to_market_config.items():
        print(f"  {ticker}: {etf}")
    if not added_to_market_config:
        print("  (none)")

    if updated:
        print(f"\nSaved to {OVERRIDE_PATH.relative_to(REPO_ROOT)}")
    if added_to_market_config:
        print(f"Saved to {MARKET_CONFIG_PATH.relative_to(REPO_ROOT)}")
    if updated or added_to_market_config:
        print("\nChanges are not committed — review the diff and commit manually.")


if __name__ == "__main__":
    main()
