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


def add_monitoring_items(new_items: list[dict]) -> None:
    """Adds new monitoring items from synthesis output.

    Deduplicates by ticker + item text similarity — see `_similar()`.
    `new_items`: list of {ticker, item, priority, source (optional)}.
    """
    existing = _load_all()
    today_str = str(date.today())

    added = 0
    for new in new_items:
        ticker = new.get("ticker", "").upper()
        item_text = new.get("item", "")
        if not ticker or not item_text:
            continue
        duplicate = any(
            e.get("ticker") == ticker
            and e.get("status") == "active"
            and _similar(e.get("item", ""), item_text)
            for e in existing.get("items", [])
        )
        if duplicate:
            continue
        item_id = f"{ticker.lower()}_{today_str}_{added}"
        existing.setdefault("items", []).append({
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

    if added > 0:
        _save_all(existing)
        logger.info("add_monitoring_items: added %d new items", added)


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
