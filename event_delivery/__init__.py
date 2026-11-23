"""Durable local delivery of ScienceHelpy business events."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any
from uuid import uuid4

from .catalog import EVENT_STREAMS


@dataclass(frozen=True)
class Event:
    position: int
    id: str
    stream: str
    name: str
    occurred_at: str
    payload: dict[str, Any]


class EventStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS events (
                    position INTEGER PRIMARY KEY,
                    id TEXT NOT NULL UNIQUE,
                    stream TEXT NOT NULL,
                    name TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS events_by_stream ON events (stream, position);
                CREATE TABLE IF NOT EXISTS checkpoints (
                    consumer TEXT NOT NULL,
                    stream TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    PRIMARY KEY (consumer, stream)
                );
            """)

    @contextmanager
    def _connect(self):
        with closing(sqlite3.connect(self.path)) as db, db:
            yield db

    def publish(self, name: str, /, **payload: Any) -> Event:
        try:
            stream = EVENT_STREAMS[name]
        except KeyError:
            raise ValueError(f"Unknown event: {name}") from None

        event_id = str(uuid4())
        occurred_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as db:
            cursor = db.execute(
                "INSERT INTO events (id, stream, name, occurred_at, payload) "
                "VALUES (?, ?, ?, ?, ?)",
                (event_id, stream, name, occurred_at, json.dumps(payload, allow_nan=False)),
            )
            position = cursor.lastrowid
        return Event(position, event_id, stream, name, occurred_at, payload)

    def deliver(
        self, consumer: str, stream: str, handler: Callable[[Event], None], limit: int = 100
    ) -> int:
        """Deliver in stream order and checkpoint each successful handler call."""
        if not consumer.strip() or stream not in EVENT_STREAMS.values() or limit <= 0:
            raise ValueError("Expected a consumer name, a known stream and a positive limit")

        with self._connect() as db:
            rows = db.execute(
                "SELECT position, id, stream, name, occurred_at, payload FROM events "
                "WHERE stream = ? AND position > COALESCE(("
                "SELECT position FROM checkpoints WHERE consumer = ? AND stream = ?"
                "), 0) ORDER BY position LIMIT ?",
                (stream, consumer, stream, limit),
            ).fetchall()

        for position, event_id, stream, name, occurred_at, payload in rows:
            handler(Event(position, event_id, stream, name, occurred_at, json.loads(payload)))
            with self._connect() as db:
                db.execute(
                    "INSERT INTO checkpoints (consumer, stream, position) VALUES (?, ?, ?) "
                    "ON CONFLICT (consumer, stream) DO UPDATE "
                    "SET position = MAX(checkpoints.position, excluded.position)",
                    (consumer, stream, position),
                )

        return len(rows)
