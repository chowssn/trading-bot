"""Tracked, git-backed edits to equity config files.

Called by Module 3 after an AI-assisted discussion and explicit user
approval — never on its own initiative. Every write function here:

- makes the change programmatically (via `ast`-derived source spans, or
  the `positions_override.json` merge layer — never `eval`/`exec`),
- appends a dated reason to `changelog.md`,
- runs `git add` + `git commit` on the changed files,
- and never raises: on any failure it logs and returns False.

`save_thesis_update()` and `add_to_watchlist()` write to
`positions_override.json` rather than editing `positions.py`'s source —
`equity.config.positions` merges that JSON over its literal `POSITIONS`/
`WATCHLIST` dicts on import (see that module's `_load_override()`). This
sidesteps parsing/rewriting the hand-authored Python source for the two
operations that happen most often. `save_thesis_update()`,
`add_to_watchlist()`, and `remove_from_watchlist()` each call
`positions.reload()` after writing, so an already-running process sees the
change immediately — no restart or re-import needed.

`remove_from_watchlist()` and `update_market_config()` are rarer and edit
Python source directly, since there's no override layer for "field no
longer exists" or for `market_config.py`'s standalone constants. Both use
`ast` to find the exact source span of the target and replace only that
span, then re-parse the result before writing — if the edit would produce
invalid Python, nothing is written and the function returns False.
"""

import ast
import json
import logging
import re
import subprocess
from datetime import datetime
from pathlib import Path

from equity.config import positions

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CONFIG_DIR.parents[1]
_POSITIONS_PATH = _CONFIG_DIR / "positions.py"
_MARKET_CONFIG_PATH = _CONFIG_DIR / "market_config.py"
_OVERRIDE_PATH = _CONFIG_DIR / "positions_override.json"
_CHANGELOG_PATH = _CONFIG_DIR / "changelog.md"


def _current_month() -> str:
    return datetime.now().strftime("%Y-%m")


# ---------------------------------------------------------------------------
# git plumbing
# ---------------------------------------------------------------------------

def _git_commit(paths: list[Path], message: str) -> bool:
    """`git add` the given paths and commit them with `message`. Never raises."""
    try:
        rel_paths = [str(p) for p in paths]
        add = subprocess.run(
            ["git", "add", *rel_paths], cwd=_REPO_ROOT, capture_output=True, text=True,
        )
        if add.returncode != 0:
            logger.error("git add failed for %s: %s", rel_paths, add.stderr.strip())
            return False

        commit = subprocess.run(
            ["git", "commit", "-m", message], cwd=_REPO_ROOT, capture_output=True, text=True,
        )
        if commit.returncode != 0:
            logger.error("git commit failed: %s", commit.stderr.strip())
            return False

        logger.info("Committed: %s", message.splitlines()[0])
        return True
    except OSError as exc:
        logger.error("git subprocess failed: %s", exc)
        return False


def _resolve_path(filepath: str) -> Path:
    """Resolve a filepath given as absolute, repo-relative, or config-dir-relative."""
    path = Path(filepath)
    if path.is_absolute():
        return path
    candidate = _REPO_ROOT / path
    if candidate.exists():
        return candidate
    candidate = _CONFIG_DIR / path
    if candidate.exists():
        return candidate
    return _REPO_ROOT / path  # let git report "no such path" if it's genuinely missing


# ---------------------------------------------------------------------------
# positions_override.json (used by save_thesis_update, add_to_watchlist)
# ---------------------------------------------------------------------------

def _load_override_raw() -> dict:
    if not _OVERRIDE_PATH.exists():
        return {"POSITIONS": {}, "WATCHLIST": {}}
    try:
        with open(_OVERRIDE_PATH) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("positions_override.json unreadable, starting fresh: %s", exc)
        return {"POSITIONS": {}, "WATCHLIST": {}}
    data.setdefault("POSITIONS", {})
    data.setdefault("WATCHLIST", {})
    return data


def _write_override(data: dict) -> None:
    with open(_OVERRIDE_PATH, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


# ---------------------------------------------------------------------------
# ast-based source surgery (used by remove_from_watchlist, update_market_config)
# ---------------------------------------------------------------------------

def _line_offsets(source: str) -> list[int]:
    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def _module_dict_keys(source: str, dict_name: str) -> set[str]:
    """String keys of the module-level `dict_name = {...}` dict literal, if any."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Dict)
            and any(isinstance(t, ast.Name) and t.id == dict_name for t in node.targets)
        ):
            return {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
    return set()


def _remove_dict_key(source: str, dict_name: str, key: str) -> str | None:
    """Remove the `"key": {...}` entry from module-level dict `dict_name`.

    Returns the edited source, or None if `key` wasn't found in `dict_name`
    or the edit would produce invalid Python (nothing is written either way
    — the caller decides what None means). Any comment lines immediately
    above the removed entry are left in place; this is a cosmetic
    limitation, not a correctness one.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        logger.error("_remove_dict_key: source does not parse: %s", exc)
        return None

    span = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Dict)
            and any(isinstance(t, ast.Name) and t.id == dict_name for t in node.targets)
        ):
            for k_node, v_node in zip(node.value.keys, node.value.values):
                if isinstance(k_node, ast.Constant) and k_node.value == key:
                    span = (k_node.lineno, v_node.end_lineno, v_node.end_col_offset)
                    break
            break

    if span is None:
        logger.error("_remove_dict_key: %r not found in %s", key, dict_name)
        return None

    start_line, end_line, end_col = span
    offsets = _line_offsets(source)
    start = offsets[start_line - 1]  # back up to the start of the key's own line
    end = offsets[end_line - 1] + end_col

    # Consume a trailing comma and the remainder of that line.
    m = re.match(r"[ \t]*,?[ \t]*\r?\n?", source[end:])
    end += m.end()

    new_source = source[:start] + source[end:]

    try:
        ast.parse(new_source)
    except SyntaxError as exc:
        logger.error("_remove_dict_key: edit would produce invalid Python, aborting: %s", exc)
        return None

    return new_source


def _set_field_path(source: str, field_path: str, new_value) -> str | None:
    """Replace the value at `field_path` (e.g. 'VIX_ELEVATED' or 'REGIME_RULES.HIGH_VOL').

    The first segment must be a module-level assignment target; each further
    segment descends into a string key of a dict literal. Returns the edited
    source, or None if the path doesn't resolve or the edit would produce
    invalid Python.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        logger.error("_set_field_path: source does not parse: %s", exc)
        return None

    parts = field_path.split(".")
    target_node = None

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == parts[0] for t in node.targets
        ):
            value_node = node.value
            for part in parts[1:]:
                if not isinstance(value_node, ast.Dict):
                    value_node = None
                    break
                next_node = None
                for k_node, v_node in zip(value_node.keys, value_node.values):
                    if isinstance(k_node, ast.Constant) and k_node.value == part:
                        next_node = v_node
                        break
                value_node = next_node
                if value_node is None:
                    break
            target_node = value_node
            break

    if target_node is None:
        logger.error("_set_field_path: %r not found", field_path)
        return None

    offsets = _line_offsets(source)
    start = offsets[target_node.lineno - 1] + target_node.col_offset
    end = offsets[target_node.end_lineno - 1] + target_node.end_col_offset

    new_source = source[:start] + repr(new_value) + source[end:]

    try:
        ast.parse(new_source)
    except SyntaxError as exc:
        logger.error("_set_field_path: edit would produce invalid Python, aborting: %s", exc)
        return None

    return new_source


# ---------------------------------------------------------------------------
# changelog
# ---------------------------------------------------------------------------

def append_changelog(entry: str) -> None:
    """Append a dated bullet to config/changelog.md.

    Entries made the same day are grouped under one `## YYYY-MM-DD` header.
    Never raises — logs and gives up silently on any I/O error.
    """
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        header = f"## {today}"
        bullet = f"- {entry}\n"

        text = _CHANGELOG_PATH.read_text() if _CHANGELOG_PATH.exists() else "# Config Changelog\n"

        if header in text:
            idx = text.index(header)
            next_header_idx = text.find("\n## ", idx + len(header))
            insert_at = next_header_idx if next_header_idx != -1 else len(text)
            text = text[:insert_at].rstrip("\n") + "\n" + bullet + text[insert_at:]
        else:
            if not text.endswith("\n"):
                text += "\n"
            text += f"\n{header}\n{bullet}"

        _CHANGELOG_PATH.write_text(text)
    except OSError as exc:
        logger.error("append_changelog failed: %s", exc)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def save_thesis_update(ticker: str, updates: dict, reason: str) -> bool:
    """Update thesis fields for `ticker` via positions_override.json and commit.

    Called by Module 3 after AI discussion and user approval. `updates` is a
    dict of fields to change, e.g. {'thesis': '...', 'thesis_breakers': [...]}.
    `reason` is a human-readable explanation, recorded in the git commit and
    changelog. Also bumps the ticker's `last_reviewed` to the current month.
    Returns True on success, False (logged) on any failure.
    """
    try:
        if ticker in positions.POSITIONS:
            section = "POSITIONS"
        elif ticker in positions.WATCHLIST:
            section = "WATCHLIST"
        else:
            logger.error("save_thesis_update: %s not found in POSITIONS or WATCHLIST", ticker)
            return False

        override = _load_override_raw()
        override[section].setdefault(ticker, {})
        override[section][ticker].update(updates)
        override[section][ticker]["last_reviewed"] = _current_month()
        _write_override(override)

        append_changelog(f"positions.py: Updated {ticker} thesis — {reason}")

        result = _git_commit(
            [_OVERRIDE_PATH, _CHANGELOG_PATH],
            f"config: update {ticker} thesis\n\n{reason}",
        )
        positions.reload()
        return result
    except Exception as exc:
        logger.error("save_thesis_update failed for %s: %s", ticker, exc)
        return False


def update_position_tier(ticker: str, tier_v2: str, style: str,
                          classification_status: str, reason: str) -> bool:
    """Update tier_v2/style/classification_status for a position and commit.

    Writes through `save_thesis_update()` (positions_override.json merge
    layer), so this shares its git-commit + changelog + reload behavior.
    Called after an advisor discussion leads to reclassification under the
    POSITION_TIERS framework (see market_config.py). Returns True on
    success, False (logged) on any failure.
    """
    ticker = ticker.upper()
    updates = {
        "tier_v2": tier_v2,
        "style": style,
        "classification_status": classification_status,
    }
    result = save_thesis_update(ticker, updates, reason)
    if result:
        append_changelog(
            f"{ticker} reclassified to {tier_v2} ({style}) — {reason}"
        )
    return result


def add_to_watchlist(ticker: str, position_dict: dict, reason: str) -> bool:
    """Add a new ticker to WATCHLIST via positions_override.json and commit.

    Called after AI discussion and user approval. Returns False (logged)
    if `ticker` already exists in POSITIONS or WATCHLIST, or on any error.
    """
    try:
        if ticker in positions.POSITIONS or ticker in positions.WATCHLIST:
            logger.error("add_to_watchlist: %s already exists", ticker)
            return False

        override = _load_override_raw()
        entry = dict(position_dict)
        entry.setdefault("thesis_source", "ai_assisted")
        entry.setdefault("last_reviewed", _current_month())
        override["WATCHLIST"][ticker] = entry
        _write_override(override)

        append_changelog(f"positions.py: {ticker} added to watchlist — {reason}")

        result = _git_commit(
            [_OVERRIDE_PATH, _CHANGELOG_PATH],
            f"config: add {ticker} to watchlist\n\n{reason}",
        )
        positions.reload()
        return result
    except Exception as exc:
        logger.error("add_to_watchlist failed for %s: %s", ticker, exc)
        return False


def remove_from_watchlist(ticker: str, reason: str) -> bool:
    """Remove `ticker` from WATCHLIST and commit.

    Removes it from positions_override.json if it was added there (covers
    tickers added via `add_to_watchlist()`); if it's defined directly in
    positions.py's source WATCHLIST dict, edits the source file instead
    (via `ast` — see `_remove_dict_key()`). Returns False (logged) if
    `ticker` isn't found in either place, or on any error.
    """
    try:
        removed = False
        changed_paths = []

        override = _load_override_raw()
        if ticker in override["WATCHLIST"]:
            del override["WATCHLIST"][ticker]
            _write_override(override)
            changed_paths.append(_OVERRIDE_PATH)
            removed = True

        base_source = _POSITIONS_PATH.read_text()
        if ticker in _module_dict_keys(base_source, "WATCHLIST"):
            new_source = _remove_dict_key(base_source, "WATCHLIST", ticker)
            if new_source is not None:
                _POSITIONS_PATH.write_text(new_source)
                changed_paths.append(_POSITIONS_PATH)
                removed = True
            elif not removed:
                logger.error("remove_from_watchlist: could not edit positions.py for %s", ticker)
                return False

        if not removed:
            logger.error("remove_from_watchlist: %s not found in WATCHLIST", ticker)
            return False

        append_changelog(f"positions.py: {ticker} removed from watchlist — {reason}")
        changed_paths.append(_CHANGELOG_PATH)

        result = _git_commit(changed_paths, f"config: remove {ticker} from watchlist\n\n{reason}")
        positions.reload()
        return result
    except Exception as exc:
        logger.error("remove_from_watchlist failed for %s: %s", ticker, exc)
        return False


def update_market_config(field_path: str, new_value, reason: str) -> bool:
    """Update a single field in market_config.py and commit.

    `field_path` is dot-notation, e.g. 'VIX_ELEVATED' for a top-level
    constant or 'REGIME_RULES.HIGH_VOL' for a key inside a top-level dict
    literal. Returns False (logged) if market_config.py doesn't exist, the
    path doesn't resolve, or on any other error.
    """
    try:
        if not _MARKET_CONFIG_PATH.exists():
            logger.error("update_market_config: %s does not exist", _MARKET_CONFIG_PATH)
            return False

        source = _MARKET_CONFIG_PATH.read_text()
        new_source = _set_field_path(source, field_path, new_value)
        if new_source is None:
            return False

        _MARKET_CONFIG_PATH.write_text(new_source)
        append_changelog(f"market_config.py: {field_path} updated to {new_value!r} — {reason}")

        return _git_commit(
            [_MARKET_CONFIG_PATH, _CHANGELOG_PATH],
            f"config: update market_config.{field_path}\n\n{reason}",
        )
    except Exception as exc:
        logger.error("update_market_config failed for %s: %s", field_path, exc)
        return False


def get_config_history(filepath: str, n_commits: int = 10) -> list[dict]:
    """Return the last `n_commits` git commits touching `filepath`.

    Each entry is {'commit', 'date', 'message', 'diff'}. Returns [] (logged)
    on any git failure, including a filepath with no history.
    """
    try:
        path = _resolve_path(filepath)
        log = subprocess.run(
            ["git", "log", f"-{n_commits}", "--format=%H%x1f%ad%x1f%s", "--date=short", "--", str(path)],
            cwd=_REPO_ROOT, capture_output=True, text=True,
        )
        if log.returncode != 0:
            logger.error("get_config_history: git log failed: %s", log.stderr.strip())
            return []

        history = []
        for line in log.stdout.strip().splitlines():
            if not line:
                continue
            commit_hash, commit_date, message = line.split("\x1f", 2)
            diff = subprocess.run(
                ["git", "show", commit_hash, "--", str(path)], cwd=_REPO_ROOT, capture_output=True, text=True,
            )
            history.append({
                "commit": commit_hash,
                "date": commit_date,
                "message": message,
                "diff": diff.stdout if diff.returncode == 0 else "",
            })
        return history
    except Exception as exc:
        logger.error("get_config_history failed for %s: %s", filepath, exc)
        return []


def revert_config(filepath: str, commit_hash: str) -> bool:
    """Revert `filepath` to its contents at `commit_hash` and commit.

    Requires explicit confirmation — this function performs the revert
    unconditionally once called, so the caller (Module 3) must obtain the
    user's explicit confirmation before invoking it. Returns False (logged)
    on any failure.
    """
    try:
        path = _resolve_path(filepath)
        rel_path = path.relative_to(_REPO_ROOT) if path.is_relative_to(_REPO_ROOT) else path

        show = subprocess.run(
            ["git", "show", f"{commit_hash}:{rel_path}"], cwd=_REPO_ROOT, capture_output=True, text=True,
        )
        if show.returncode != 0:
            logger.error("revert_config: could not read %s at %s: %s", rel_path, commit_hash, show.stderr.strip())
            return False

        path.write_text(show.stdout)
        append_changelog(f"{path.name}: reverted to {commit_hash[:8]}")

        return _git_commit([path, _CHANGELOG_PATH], f"config: revert {path.name} to {commit_hash[:8]}")
    except Exception as exc:
        logger.error("revert_config failed for %s at %s: %s", filepath, commit_hash, exc)
        return False
