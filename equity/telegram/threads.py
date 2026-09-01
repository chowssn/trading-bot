"""SQLite-backed thread management for the Telegram portfolio advisor.

Threads are per-ticker or per-topic conversations with Claude. Full
message history is always preserved in the `messages` table; the
`recent_verbatim` windowing in `get_messages_for_api()` only affects what
gets sent to the Claude API on a given call, never what's stored.

Database at `equity/data/advisor.db` (gitignored — see .gitignore).
"""

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = "equity/data/advisor.db"


class ThreadManager:
    def __init__(self, db_path: str = _DEFAULT_DB_PATH):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.initialize_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def initialize_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS threads (
                    thread_id TEXT PRIMARY KEY,
                    thread_type TEXT NOT NULL,
                    subject TEXT,
                    created_at TEXT NOT NULL,
                    last_active TEXT NOT NULL,
                    summary TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    summarized INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY (thread_id) REFERENCES threads(thread_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pending_changes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    change_type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    description TEXT,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS active_thread (
                    user_id INTEGER PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_id)")
            conn.commit()

    # ------------------------------------------------------------------
    # Threads / messages
    # ------------------------------------------------------------------

    def get_or_create_thread(self, thread_id: str, thread_type: str, subject: str) -> dict:
        now = datetime.now().isoformat()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM threads WHERE thread_id = ?", (thread_id,)
            ).fetchone()
            if row is not None:
                return dict(row)
            conn.execute(
                """
                INSERT INTO threads (thread_id, thread_type, subject, created_at, last_active, summary)
                VALUES (?, ?, ?, ?, ?, NULL)
                """,
                (thread_id, thread_type, subject, now, now),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM threads WHERE thread_id = ?", (thread_id,)
            ).fetchone()
            return dict(row)

    def add_message(self, thread_id: str, role: str, content: str) -> int:
        now = datetime.now().isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO messages (thread_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                (thread_id, role, content, now),
            )
            conn.execute(
                "UPDATE threads SET last_active = ? WHERE thread_id = ?",
                (now, thread_id),
            )
            conn.commit()
            message_id = cur.lastrowid

        # Auto-summarization needs Advisor.summarize_messages (an API call),
        # which would create a circular import if wired in here. Advisor.chat()
        # calls auto_summarize_thread() explicitly after each add_message —
        # see advisor.py — so the check still runs on every turn.
        return message_id

    def get_messages_for_api(self, thread_id: str, recent_verbatim: int = 50) -> list[dict]:
        with self._connect() as conn:
            thread_row = conn.execute(
                "SELECT summary FROM threads WHERE thread_id = ?", (thread_id,)
            ).fetchone()
            summary = thread_row["summary"] if thread_row else None

            rows = conn.execute(
                """
                SELECT role, content FROM messages
                WHERE thread_id = ? AND summarized = 0
                ORDER BY id ASC
                """,
                (thread_id,),
            ).fetchall()

        recent = [{"role": r["role"], "content": r["content"]} for r in rows][-recent_verbatim:]

        if summary:
            note = {
                "role": "user",
                "content": (
                    f"[Earlier conversation summary — context only, not a new "
                    f"message to respond to directly: {summary}]"
                ),
            }
            return [note] + recent
        return recent

    def get_thread_info(self, thread_id: str) -> dict | None:
        with self._connect() as conn:
            thread_row = conn.execute(
                "SELECT * FROM threads WHERE thread_id = ?", (thread_id,)
            ).fetchone()
            if thread_row is None:
                return None
            count_row = conn.execute(
                "SELECT COUNT(*) AS n FROM messages WHERE thread_id = ?", (thread_id,)
            ).fetchone()

        return {
            "thread_id": thread_id,
            "thread_type": thread_row["thread_type"],
            "subject": thread_row["subject"],
            "message_count": count_row["n"],
            "last_active": thread_row["last_active"],
            "has_summary": thread_row["summary"] is not None,
        }

    def list_threads(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM threads ORDER BY last_active DESC"
            ).fetchall()
            result = []
            for row in rows:
                count_row = conn.execute(
                    "SELECT COUNT(*) AS n FROM messages WHERE thread_id = ?",
                    (row["thread_id"],),
                ).fetchone()
                d = dict(row)
                d["message_count"] = count_row["n"]
                result.append(d)
            return result

    # ------------------------------------------------------------------
    # Active thread per user
    # ------------------------------------------------------------------

    def set_active_thread(self, user_id: int, thread_id: str) -> None:
        now = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO active_thread (user_id, thread_id, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET thread_id = excluded.thread_id,
                    updated_at = excluded.updated_at
                """,
                (user_id, thread_id, now),
            )
            conn.commit()

    def get_active_thread(self, user_id: int) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT thread_id FROM active_thread WHERE user_id = ?", (user_id,)
            ).fetchone()
            return row["thread_id"] if row else None

    def clear_active_thread(self, user_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM active_thread WHERE user_id = ?", (user_id,))
            conn.commit()

    # ------------------------------------------------------------------
    # Auto-summarization
    # ------------------------------------------------------------------

    def auto_summarize_thread(self, thread_id: str, summarize_fn) -> None:
        with self._connect() as conn:
            unsummarized = conn.execute(
                "SELECT COUNT(*) AS n FROM messages WHERE thread_id = ? AND summarized = 0",
                (thread_id,),
            ).fetchone()["n"]

            if unsummarized <= 100:
                return

            oldest = conn.execute(
                """
                SELECT id, role, content FROM messages
                WHERE thread_id = ? AND summarized = 0
                ORDER BY id ASC LIMIT 80
                """,
                (thread_id,),
            ).fetchall()

        if not oldest:
            return

        try:
            messages_for_summary = [{"role": r["role"], "content": r["content"]} for r in oldest]
            new_summary = summarize_fn(messages_for_summary)
        except Exception as exc:
            logger.error("auto_summarize_thread: summarize_fn failed for %s: %s", thread_id, exc)
            return

        ids = [r["id"] for r in oldest]
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT summary FROM threads WHERE thread_id = ?", (thread_id,)
            ).fetchone()
            combined = (
                f"{existing['summary']}\n\n{new_summary}"
                if existing and existing["summary"]
                else new_summary
            )
            conn.execute(
                "UPDATE threads SET summary = ? WHERE thread_id = ?",
                (combined, thread_id),
            )
            conn.executemany(
                "UPDATE messages SET summarized = 1 WHERE id = ?",
                [(i,) for i in ids],
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Pending changes (config edits awaiting /confirm)
    # ------------------------------------------------------------------

    def save_pending_change(
        self, change_type: str, payload: dict, description: str, expires_hours: float = 0.5
    ) -> int:
        now = datetime.now()
        expires_at = now + timedelta(hours=expires_hours)
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO pending_changes (change_type, payload, description, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (change_type, json.dumps(payload), description, now.isoformat(), expires_at.isoformat()),
            )
            conn.commit()
            return cur.lastrowid

    def _row_to_change(self, row: sqlite3.Row) -> dict | None:
        if row is None:
            return None
        expires_at = datetime.fromisoformat(row["expires_at"])
        if datetime.now() > expires_at:
            return None
        return {
            "id": row["id"],
            "change_type": row["change_type"],
            "payload": json.loads(row["payload"]),
            "description": row["description"],
            "created_at": row["created_at"],
            "expires_at": row["expires_at"],
        }

    def get_pending_change(self, change_id: int) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM pending_changes WHERE id = ?", (change_id,)
            ).fetchone()
            return self._row_to_change(row)

    def get_latest_pending_change(self) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM pending_changes ORDER BY id DESC LIMIT 1"
            ).fetchone()
            return self._row_to_change(row)

    def clear_pending_change(self, change_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM pending_changes WHERE id = ?", (change_id,))
            conn.commit()

    def clear_latest_pending_change(self) -> None:
        latest = self.get_latest_pending_change()
        if latest:
            self.clear_pending_change(latest["id"])

    def clear_expired_pending_changes(self) -> None:
        now = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute("DELETE FROM pending_changes WHERE expires_at < ?", (now,))
            conn.commit()
