"""Persistent monitoring list for the morning brief.

Items are added by `equity.brief.brief_synthesizer.synthesize_performance()`
(via `_parse_and_persist_monitoring()`) and dismissed explicitly by the
user — via the Telegram `/dismiss` command/buttons, or through advisor
discussion nudging the user toward `/dismiss`.

Storage: equity/data/monitoring.json (gitignored — see .gitignore).
Schema:
{
  "items": [
    {
      "id": "tsla_2026-09-04_0",
      "ticker": "TSLA",
      "item": "Megapack margin trajectory in next earnings",
      "priority": "high",
      "added_date": "2026-09-04",
      "added_from": "performance_synthesis",
      "status": "active",  # active | resolved | dismissed | escalated
      "last_checked": "2026-09-04",
      "notes": []
    }
  ]
}
"""

import json
import logging
import re
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

MONITORING_PATH = Path(__file__).resolve().parent / "monitoring.json"


def load_monitoring() -> list[dict]:
    """Returns all active monitoring items, each annotated with `age_days`."""
    all_data = _load_all()
    items = all_data.get("items", [])
    today = date.today()
    for item in items:
        try:
            added = date.fromisoformat(item.get("added_date", str(today)))
            item["age_days"] = (today - added).days
        except ValueError:
            item["age_days"] = 0
    return [i for i in items if i.get("status") == "active"]


# A token is a plausible ticker or macro-label word: letters/digits plus
# `.`/`-`, 1-10 chars. Deliberately allows a leading digit ("10Y", "30Y")
# for tenor labels — see `_extract_ticker()`.
_TICKER_TOKEN_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,9}$")


def _extract_ticker(raw_ticker: str) -> list[str]:
    """Extracts one or more valid ticker/macro-label symbols from a raw
    ticker-cell string, tolerating the malformed shapes synthesis output
    has actually produced (see monitoring.json history before this fix):

      "[FCX]"          -> ["FCX"]            bracket wrapper
      "COPPER $6.87"   -> ["COPPER"]         price appended
      "[CCJ/CEG]"      -> ["CCJ", "CEG"]     two tickers merged
      "FCX, CAT"       -> ["FCX", "CAT"]     comma-separated
      "10Y YIELD"      -> ["10Y YIELD"]      legitimate multi-word label,
                                              left intact

    Multi-word phrases are validated token-by-token rather than against a
    single-symbol pattern so real macro labels ("10Y YIELD", "30Y
    TREASURY") survive; anything longer than 3 words reads as prose that
    leaked into the ticker cell and is dropped rather than persisted.

    Returns an empty list if nothing ticker-shaped survives — callers
    should skip/resolve the item rather than persist a blank or garbled
    ticker.
    """
    # Bullet markers ("- TSLA") are stripped the same as the line-level
    # lstrip("- •*") above — a leading "-" here is a bullet, not a hyphen
    # ticker character like "BRK-B" ever starts or ends with.
    cleaned = raw_ticker.strip().strip("[]()*_-• \t")
    # Trailing price/percent annotation the model appended to the ticker
    # cell instead of the condition column ("COPPER $6.87", "VIX 22.4%").
    cleaned = re.sub(r"\s+\$[\d.]+.*$", "", cleaned)
    cleaned = re.sub(r"\s+[\d.]+%.*$", "", cleaned)

    valid = []
    for part in re.split(r"\s*[/,]\s*", cleaned):
        part = part.strip().strip("[]()*_-• ")
        if not part:
            continue
        part_upper = part.upper()
        tokens = part_upper.split()
        if not tokens or len(tokens) > 3:
            continue
        if all(_TICKER_TOKEN_RE.match(t) for t in tokens):
            valid.append(part_upper)

    return valid


def add_monitoring_items(new_items: list[dict]) -> None:
    """Adds new monitoring items from synthesis output.

    Deduplicates by ticker + item text similarity — see `_similar()`. When a
    new item supersedes an existing active item for the same ticker (an
    updated threshold, or language showing a level was reached — see
    `_supersedes()`), the old item is resolved instead of left stale
    alongside the new one.
    `new_items`: list of {ticker, item, priority, source (optional)}.
    """
    existing = _load_all()
    today_str = str(date.today())
    items = existing.setdefault("items", [])

    added = 0
    superseded = 0
    for new in new_items:
        ticker = new.get("ticker", "").upper()
        item_text = new.get("item", "")
        if not ticker or not item_text:
            continue

        ticker_active = [
            e for e in items
            if e.get("ticker") == ticker and e.get("status") == "active"
        ]

        resolved_now_ids = set()
        for existing_item in ticker_active:
            if _supersedes(item_text, existing_item.get("item", "")):
                existing_item["status"] = "resolved"
                existing_item.setdefault("notes", []).append(
                    f"Superseded {today_str} by updated item: {item_text[:60]}"
                )
                resolved_now_ids.add(id(existing_item))
                superseded += 1

        # Skip as a duplicate only if a still-active item (not one just
        # resolved above) is a near-identical repeat.
        still_active = [e for e in ticker_active if id(e) not in resolved_now_ids]
        if any(_similar(e.get("item", ""), item_text) for e in still_active):
            continue

        item_id = f"{ticker.lower()}_{today_str}_{added}"
        items.append({
            "id": item_id,
            "ticker": ticker,
            "item": item_text,
            "priority": new.get("priority", "medium"),
            "added_date": today_str,
            "added_from": new.get("source", "performance_synthesis"),
            "status": "active",
            "last_checked": today_str,
            "notes": [],
        })
        added += 1

    if added > 0 or superseded > 0:
        _save_all(existing)
        logger.info("add_monitoring_items: added %d, superseded %d", added, superseded)


def _supersedes(new_text: str, existing_text: str) -> bool:
    """True if `new_text` reads as an update/escalation of `existing_text`
    for the same monitoring topic, rather than an unrelated item:

    1. Same topic (`_similar()`) but language showing a level was reached
       ("crossed", "broke above", "now at", ...).
    2. Same topic with a different number in it (an updated threshold).
    """
    new_lower = new_text.lower()

    escalation_signals = (
        "crossed", "broke above", "broke below", "now at",
        "closed above", "closed below", "confirmed", "triggered",
    )
    same_topic = _similar(new_text, existing_text)

    if same_topic and any(sig in new_lower for sig in escalation_signals):
        return True

    new_numbers = set(re.findall(r"\d+\.?\d*", new_text))
    existing_numbers = set(re.findall(r"\d+\.?\d*", existing_text))
    if same_topic and new_numbers != existing_numbers:
        return True

    return False


def deduplicate_monitoring() -> int:
    """Resolves duplicate active items for the same ticker where a newer one
    supersedes an older one (see `_supersedes()`). Idempotent — safe to call
    on every brief run (see `brief_builder.build_morning_brief()`).

    Also normalizes malformed `ticker` fields that survived into storage
    from an earlier synthesis parse, before `_parse_and_persist_monitoring()`
    started running ticker cells through `_extract_ticker()` — without this,
    "FCX" and "[FCX]" would sit in separate buckets below and never dedupe
    against each other:

      - a fixable artifact ("[FCX]", "COPPER $6.87") is normalized in place
      - a merged ticker ("[CCJ/CEG]") is resolved and replaced by one active
        item per extracted ticker, sharing the same condition text
      - a ticker `_extract_ticker()` can't make sense of at all is resolved
        outright rather than left corrupting the monitoring list

    Only active items are touched — a resolved/dismissed item's ticker
    field is just historical record at that point.

    Returns count of items resolved.
    """
    data = _load_all()
    today_str = str(date.today())
    items = data.setdefault("items", [])
    resolved = 0
    normalized = 0

    split_items = []
    for item in items:
        if item.get("status") != "active":
            continue
        ticker = item.get("ticker", "")
        extracted = _extract_ticker(ticker)

        if not extracted:
            item["status"] = "resolved"
            item.setdefault("notes", []).append(
                f'Auto-resolved {today_str}: unparseable ticker "{ticker}"'
            )
            resolved += 1
        elif extracted == [ticker]:
            continue  # already clean
        elif len(extracted) == 1:
            item["ticker"] = extracted[0]
            normalized += 1
        else:
            item["status"] = "resolved"
            item.setdefault("notes", []).append(
                f'Auto-split {today_str}: "{ticker}" -> {extracted}'
            )
            resolved += 1
            for i, new_ticker in enumerate(extracted):
                split_items.append({
                    "id": f"{new_ticker.lower()}_{today_str}_split{i}",
                    "ticker": new_ticker,
                    "item": item["item"],
                    "priority": item.get("priority", "medium"),
                    "added_date": item.get("added_date", today_str),
                    "added_from": item.get("added_from", "split"),
                    "status": "active",
                    "last_checked": today_str,
                    "notes": [f'Split from merged ticker "{ticker}"'],
                })
            normalized += 1

    items.extend(split_items)

    by_ticker: dict[str, list[dict]] = {}
    for item in items:
        if item.get("status") == "active":
            by_ticker.setdefault(item.get("ticker", ""), []).append(item)

    for ticker, ticker_items in by_ticker.items():
        if len(ticker_items) <= 1:
            continue
        # Newest first, so a newer item can supersede an older one below it.
        ticker_items.sort(key=lambda x: x.get("added_date", ""), reverse=True)
        for i, newer in enumerate(ticker_items):
            for older in ticker_items[i + 1:]:
                if older.get("status") != "active":
                    continue
                if _supersedes(newer.get("item", ""), older.get("item", "")):
                    older["status"] = "resolved"
                    older.setdefault("notes", []).append(
                        f"Superseded {today_str} during dedup by: {newer['item'][:60]}"
                    )
                    resolved += 1

    if resolved > 0 or normalized > 0:
        _save_all(data)
        logger.info(
            "deduplicate_monitoring: normalized %d, resolved %d, split %d",
            normalized, resolved, len(split_items),
        )

    return resolved


def dismiss_monitoring(ticker: str, reason: str = "") -> int:
    """Dismisses all active monitoring items for a ticker. Returns count dismissed."""
    all_data = _load_all()
    count = 0
    for item in all_data.get("items", []):
        if item.get("ticker", "").upper() == ticker.upper() and item.get("status") == "active":
            item["status"] = "dismissed"
            item["notes"].append(f"Dismissed {date.today()}: {reason}")
            count += 1
    if count > 0:
        _save_all(all_data)
        logger.info("dismiss_monitoring: dismissed %d items for %s", count, ticker)
    return count


def dismiss_by_priority(priority: str, reason: str = "") -> int:
    """Dismisses all active monitoring items at a given priority level.
    Returns count dismissed.
    """
    all_data = _load_all()
    count = 0
    for item in all_data.get("items", []):
        if item.get("status") == "active" and item.get("priority") == priority:
            item["status"] = "dismissed"
            item["notes"].append(f"Dismissed {date.today()}: {reason}")
            count += 1
    if count > 0:
        _save_all(all_data)
        logger.info("dismiss_by_priority: dismissed %d %s-priority items", count, priority)
    return count


def dismiss_all_monitoring(reason: str = "") -> int:
    """Dismisses every active monitoring item. Returns count dismissed."""
    all_data = _load_all()
    count = 0
    for item in all_data.get("items", []):
        if item.get("status") == "active":
            item["status"] = "dismissed"
            item["notes"].append(f"Dismissed {date.today()}: {reason}")
            count += 1
    if count > 0:
        _save_all(all_data)
        logger.info("dismiss_all_monitoring: dismissed %d items", count)
    return count


def dismiss_monitoring_item(item_id: str, reason: str = "") -> bool:
    """Dismisses a specific monitoring item by ID."""
    all_data = _load_all()
    for item in all_data.get("items", []):
        if item.get("id") == item_id and item.get("status") == "active":
            item["status"] = "dismissed"
            item["notes"].append(f"Dismissed {date.today()}: {reason}")
            _save_all(all_data)
            logger.info("dismiss_monitoring_item: dismissed %s", item_id)
            return True
    return False


def get_monitoring_for_ticker(ticker: str) -> list[dict]:
    """Returns active monitoring items for a specific ticker."""
    return [i for i in load_monitoring() if i.get("ticker", "").upper() == ticker.upper()]


def _similar(a: str, b: str) -> bool:
    """Same first 30 chars, or >60% word overlap — good enough to catch a
    monitoring item the synthesizer re-proposes verbatim (or near-verbatim)
    on a later day without a full similarity library.
    """
    if a[:30].lower() == b[:30].lower():
        return True
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return False
    overlap = len(words_a & words_b) / max(len(words_a), len(words_b))
    return overlap > 0.6


def _load_all() -> dict:
    if not MONITORING_PATH.exists():
        return {"items": []}
    try:
        with open(MONITORING_PATH) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("monitoring.json unreadable, treating as empty: %s", exc)
        return {"items": []}


def _save_all(data: dict) -> None:
    MONITORING_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(MONITORING_PATH, "w") as f:
            json.dump(data, f, indent=2)
    except OSError as exc:
        logger.warning("Failed to write %s: %s", MONITORING_PATH, exc)
