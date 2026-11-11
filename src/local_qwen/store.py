from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ConversationStore:
    """SQLite-backed chat history and explicitly saved local memories."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    tier TEXT NOT NULL,
                    title TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS messages_by_session
                    ON messages(session_id, id);
                CREATE TABLE IF NOT EXISTS memories (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def create_session(self, tier: str) -> str:
        session_id = uuid.uuid4().hex[:12]
        now = _now()
        with self._connect() as db:
            db.execute(
                "INSERT INTO sessions(id, tier, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (session_id, tier.lower(), now, now),
            )
        return session_id

    def session(self, session_id: str) -> dict | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return dict(row) if row else None

    def list_sessions(self, limit: int = 20) -> list[dict]:
        with self._connect() as db:
            rows = db.execute(
                """SELECT s.*, COUNT(m.id) AS message_count
                   FROM sessions s LEFT JOIN messages m ON m.session_id = s.id
                   GROUP BY s.id ORDER BY s.updated_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def append_turn(self, session_id: str, user_text: str, assistant_text: str) -> None:
        now = _now()
        with self._connect() as db:
            row = db.execute("SELECT title FROM sessions WHERE id = ?", (session_id,)).fetchone()
            if row is None:
                raise KeyError(f"unknown conversation session {session_id}")
            title = row["title"] or " ".join(user_text.split())[:100]
            db.executemany(
                "INSERT INTO messages(session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                [
                    (session_id, "user", user_text, now),
                    (session_id, "assistant", assistant_text, now),
                ],
            )
            db.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                (title, now, session_id),
            )

    def history(self, session_id: str, limit: int = 40) -> list[dict[str, str]]:
        with self._connect() as db:
            rows = db.execute(
                """SELECT role, content FROM (
                       SELECT id, role, content FROM messages
                       WHERE session_id = ? ORDER BY id DESC LIMIT ?
                   ) ORDER BY id""",
                (session_id, limit),
            ).fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in rows]

    def clear_history(self, session_id: str) -> None:
        with self._connect() as db:
            db.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            db.execute("UPDATE sessions SET title = '', updated_at = ? WHERE id = ?", (_now(), session_id))

    def remember(self, key: str, value: str) -> None:
        key, value = key.strip(), value.strip()
        if not key or not value:
            raise ValueError("memory key and value must both be non-empty")
        now = _now()
        with self._connect() as db:
            db.execute(
                """INSERT INTO memories(key, value, created_at, updated_at) VALUES (?, ?, ?, ?)
                   ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at""",
                (key, value, now, now),
            )

    def forget(self, key: str) -> bool:
        with self._connect() as db:
            cursor = db.execute("DELETE FROM memories WHERE key = ?", (key.strip(),))
        return cursor.rowcount > 0

    def memories(self) -> list[dict[str, str]]:
        with self._connect() as db:
            rows = db.execute("SELECT key, value FROM memories ORDER BY key COLLATE NOCASE").fetchall()
        return [{"key": row["key"], "value": row["value"]} for row in rows]
