"""SQLite storage for article-specific chat threads."""

from contextlib import closing, contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from uuid import uuid4


class ChatConflict(RuntimeError):
    """A newer turn was saved after the current thread was read."""


class ChatStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS threads (
                    id TEXT PRIMARY KEY,
                    owner TEXT NOT NULL,
                    paper TEXT NOT NULL,
                    title TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    revision INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS threads_by_paper
                    ON threads (owner, paper, updated_at DESC);
                CREATE TABLE IF NOT EXISTS turns (
                    thread_id TEXT NOT NULL REFERENCES threads(id),
                    position INTEGER NOT NULL,
                    question TEXT NOT NULL,
                    reply TEXT NOT NULL,
                    PRIMARY KEY (thread_id, position)
                );
            """)

    @contextmanager
    def _connect(self):
        with closing(sqlite3.connect(self.path)) as db, db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys = ON")
            yield db

    def list(self, owner: str, paper: str) -> list[dict]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT id, title, updated_at, revision FROM threads "
                "WHERE owner = ? AND paper = ? ORDER BY updated_at DESC, id",
                (owner, paper),
            ).fetchall()
        return [dict(row) for row in rows]

    def create(self, owner: str, paper: str) -> dict:
        thread = {"id": uuid4().hex, "title": "Новый чат", "updated_at": now(), "revision": 0}
        with self._connect() as db:
            db.execute(
                "INSERT INTO threads (id, owner, paper, title, updated_at) VALUES (?, ?, ?, ?, ?)",
                (thread["id"], owner, paper, thread["title"], thread["updated_at"]),
            )
        return {**thread, "turns": []}

    def get(self, owner: str, paper: str, thread_id: str, recent: int | None = None) -> dict | None:
        with self._connect() as db:
            db.execute("BEGIN")
            thread = db.execute(
                "SELECT id, title, updated_at, revision FROM threads "
                "WHERE owner = ? AND paper = ? AND id = ?",
                (owner, paper, thread_id),
            ).fetchone()
            if thread is None:
                return None
            if recent is None:
                rows = db.execute(
                    "SELECT question, reply FROM turns WHERE thread_id = ? ORDER BY position",
                    (thread_id,),
                ).fetchall()
            else:
                rows = db.execute(
                    "SELECT question, reply FROM turns WHERE thread_id = ? "
                    "ORDER BY position DESC LIMIT ?",
                    (thread_id, recent),
                ).fetchall()[::-1]
        return {**thread, "turns": [decode_turn(row) for row in rows]}

    def append(self, owner: str, paper: str, thread_id: str, revision: int,
               question: str, reply: dict) -> dict:
        title = " ".join(question.split())
        if len(title) > 76:
            title = title[:75] + "…"
        updated_at = now()
        with self._connect() as db:
            row = db.execute(
                "UPDATE threads SET revision = revision + 1, "
                "title = CASE WHEN revision = 0 THEN ? ELSE title END, updated_at = ? "
                "WHERE owner = ? AND paper = ? AND id = ? AND revision = ? "
                "RETURNING id, title, updated_at, revision",
                (title, updated_at, owner, paper, thread_id, revision),
            ).fetchone()
            if row is None:
                raise ChatConflict
            db.execute(
                "INSERT INTO turns (thread_id, position, question, reply) VALUES (?, ?, ?, ?)",
                (thread_id, row["revision"], question, json.dumps(reply, ensure_ascii=False)),
            )
        return {**row, "turn": {"question": question, **reply}}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def decode_turn(row) -> dict:
    return {"question": row["question"], **json.loads(row["reply"])}
