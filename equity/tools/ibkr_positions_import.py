"""One-off importer: IBKR positions.csv -> equity/config/positions_override.json.

IBKR's "export positions" report is tab-separated, not comma-separated, and
its column headers/values carry stray non-breaking spaces. This script reads
that raw export, drops non-equity rows (options, VIX/index derivatives,
ForecastEx prediction-market contracts), and writes size/cost data into
`positions_override.json` — merged onto existing theses where one already
exists in `positions.py`, or as a blank-thesis POSITIONS template for
tickers we don't have a thesis on yet. A ticker only stays under WATCHLIST
if it already carries a real (non-import) thesis there in `positions.py`
— e.g. APP.

This is a stopgap until the full IBKR API (Module 4) is connected; run it
manually after each CSV export. It only ever touches
`positions_override.json`, never `positions.py` itself — see
`equity.config.positions._load_override()` for how the two are merged at
import time.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from equity.config import positions

REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = REPO_ROOT / "equity" / "data" / "ibkr" / "positions.csv"
OVERRIDE_PATH = REPO_ROOT / "equity" / "config" / "positions_override.json"

NUMERIC_COLUMNS = ["% of Net Liq", "Position", "Cost Basis", "Market Value", "Avg Price", "Last"]

LAST_REVIEWED = "2026-09"
IBKR_IMPORT_THESIS = "Imported from IBKR — update via /update TICKER"

# Option contract dates like "Jan15'27" or "Oct20'26".
_DATE_PATTERN = re.compile(r"[A-Z][a-z]{2}\d{1,2}'\d{2}")
# Option strike prices: a number followed by whitespace, e.g. "640 Put".
_STRIKE_PATTERN = re.compile(r"\d+(?:\.\d+)?\s")
# Options / prediction-market keywords. Word-bounded so real tickers that
# happen to contain these letters (e.g. "NOW") aren't caught.
_DERIVATIVE_KEYWORDS = re.compile(r"\b(Put|Call|YES|NO)\b")
_FORECASTX_MARKER = "@FORECASTX"
_TICKER_PATTERN = re.compile(r"^[A-Z]+")


def _clean_numeric(value: object) -> float:
    cleaned = str(value).strip().replace("%", "").replace("$", "").replace(",", "")
    return float(cleaned)


def _is_blank_or_total(instrument: str) -> bool:
    return instrument == "" or instrument == "Total" or "--" in instrument


def _derivative_reason(instrument: str) -> str | None:
    """Return why `instrument` is a non-equity row, or None if it's an equity."""
    if _FORECASTX_MARKER in instrument:
        return "prediction market"
    if _DERIVATIVE_KEYWORDS.search(instrument) or _DATE_PATTERN.search(instrument) or _STRIKE_PATTERN.search(instrument):
        return "options position"
    return None


def _extract_ticker(instrument: str) -> str | None:
    first_token = instrument.split(" ", 1)[0]
    match = _TICKER_PATTERN.match(first_token)
    return match.group(0) if match else None


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
    return df


def parse_positions(df: pd.DataFrame) -> tuple[dict, list[str], list[tuple[str, str]]]:
    """Returns (imported {ticker: fields}, blank/total rows, [(instrument, reason)] derivative rows)."""
    imported: dict = {}
    skipped_blank: list[str] = []
    skipped_derivative: list[tuple[str, str]] = []

    for _, row in df.iterrows():
        instrument = str(row.get("Instrument", "")).strip()

        if _is_blank_or_total(instrument):
            skipped_blank.append(instrument)
            continue

        reason = _derivative_reason(instrument)
        if reason is not None:
            skipped_derivative.append((instrument, reason))
            continue

        ticker = _extract_ticker(instrument)
        if not ticker:
            skipped_blank.append(instrument)
            continue

        try:
            numeric = {col: _clean_numeric(row[col]) for col in NUMERIC_COLUMNS}
        except (KeyError, ValueError):
            skipped_blank.append(instrument)
            continue

        imported[ticker] = {
            "avg_cost": numeric["Avg Price"],
            "shares": numeric["Position"],
            "size_pct": numeric["% of Net Liq"],
            "market_value": numeric["Market Value"],
            "tier": "core",
            "thesis": IBKR_IMPORT_THESIS,
            "thesis_breakers": [],
            "macro_thesis": "",
            "target_exit_conditions": "",
            "sector": "",
            "thesis_source": "ibkr_import",
            "last_reviewed": LAST_REVIEWED,
            "peer_tickers": [],
        }

    return imported, skipped_blank, skipped_derivative


def _has_real_thesis(entry: dict) -> bool:
    thesis = entry.get("thesis")
    return bool(thesis) and thesis != IBKR_IMPORT_THESIS and entry.get("thesis_source") != "ibkr_import"


def _load_override_json() -> dict:
    if not OVERRIDE_PATH.exists():
        return {"POSITIONS": {}, "WATCHLIST": {}}
    try:
        with open(OVERRIDE_PATH) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        data = {}
    return {"POSITIONS": data.get("POSITIONS", {}), "WATCHLIST": data.get("WATCHLIST", {})}


def merge_into_override(imported: dict) -> tuple[dict, list[str], list[str]]:
    """Merge imported positions into the override JSON.

    Tickers with a real thesis already in `positions.py` (POSITIONS or
    WATCHLIST) get only their size/cost fields updated — thesis fields are
    left untouched, and the ticker stays in whichever section already had
    it. New tickers (no real thesis anywhere) are actual IBKR holdings, so
    they get the full blank-thesis template filed under POSITIONS.

    Returns (override_json, preserved_tickers, added_tickers).
    """
    override_json = _load_override_json()

    preserved: list[str] = []
    added: list[str] = []

    for ticker, fields in imported.items():
        numeric_only = {
            "avg_cost": fields["avg_cost"],
            "shares": fields["shares"],
            "size_pct": fields["size_pct"],
            "market_value": fields["market_value"],
        }

        if ticker in positions.POSITIONS and _has_real_thesis(positions.POSITIONS[ticker]):
            section = "POSITIONS"
            entry = numeric_only
            preserved.append(ticker)
        elif ticker in positions.WATCHLIST and _has_real_thesis(positions.WATCHLIST[ticker]):
            section = "WATCHLIST"
            entry = numeric_only
            preserved.append(ticker)
        else:
            # A ticker IBKR reports us as holding is an actual position, not
            # a watchlist name — file it under POSITIONS unless it already
            # has a real thesis parked in positions.py's WATCHLIST (handled
            # above), e.g. APP.
            section = "POSITIONS"
            entry = fields
            added.append(ticker)

        override_json[section][ticker] = {**override_json[section].get(ticker, {}), **entry}

    # Present "already has a thesis" tickers in positions.py's own dict
    # order rather than CSV order.
    preserved_order = [t for t in positions.POSITIONS if t in preserved]
    preserved_order += [t for t in positions.WATCHLIST if t in preserved and t not in preserved_order]

    return override_json, preserved_order, added


def save_override(override_json: dict) -> None:
    with open(OVERRIDE_PATH, "w") as f:
        json.dump(override_json, f, indent=2, sort_keys=True)
        f.write("\n")


def main() -> None:
    df = load_csv(CSV_PATH)
    imported, skipped_blank, skipped_derivative = parse_positions(df)
    override_json, preserved, added = merge_into_override(imported)
    save_override(override_json)

    print(f"Imported {len(imported)} positions from IBKR CSV")
    print(f"Already in positions.py (thesis preserved): {', '.join(preserved) if preserved else '(none)'}")
    print(f"Added to override with size/cost data: {', '.join(added) if added else '(none)'}")
    print(f"Skipped (blank/total rows): {len(skipped_blank)}")
    print("Skipped (options/derivatives/prediction markets):")
    for instrument, reason in skipped_derivative:
        print(f"  {instrument} — {reason}")
    print(f"Saved to {OVERRIDE_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
