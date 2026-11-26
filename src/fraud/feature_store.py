"""Online SQLite feature store (M10).

Two tables, one JSON document per key:

    uid_state(key TEXT PRIMARY KEY, state TEXT)    -- history.UidState
    card1_state(key TEXT PRIMARY KEY, state TEXT)  -- history.CardState

JSON keeps floats exactly, so stored state gives the same features as the batch pass.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from fraud.features.history import CardState, HistoryStates, UidState


class FeatureStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        # Losing the last write in a power cut is acceptable; per-request latency is not.
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        for table in ("uid_state", "card1_state"):
            self._db.execute(
                f"CREATE TABLE IF NOT EXISTS {table} (key TEXT PRIMARY KEY, state TEXT NOT NULL)"
            )
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    def _read(self, table: str, key: str) -> dict | None:
        row = self._db.execute(f"SELECT state FROM {table} WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def _write(self, table: str, key: str, state: dict) -> None:
        self._db.execute(
            f"INSERT OR REPLACE INTO {table} (key, state) VALUES (?, ?)",
            (key, json.dumps(state)),
        )

    @contextmanager
    def transaction(self, uid: str, card1: str) -> Iterator[tuple[UidState, CardState]]:
        """Read both states, let the caller update them, and save them atomically.

        Nothing is saved if the caller raises.
        """
        with self._lock:
            stored_uid = self._read("uid_state", uid)
            stored_card = self._read("card1_state", card1)
            uid_state = UidState(**stored_uid) if stored_uid else UidState()
            card_state = CardState(**stored_card) if stored_card else CardState()
            yield uid_state, card_state
            with self._db:
                self._write("uid_state", uid, uid_state.to_dict())
                self._write("card1_state", card1, card_state.to_dict())

    def load(self, states: HistoryStates) -> None:
        """Replace the store contents with warmed state."""
        with self._lock, self._db:
            self._db.execute("DELETE FROM uid_state")
            self._db.execute("DELETE FROM card1_state")
            for uid, uid_state in states.uids.items():
                self._write("uid_state", uid, uid_state.to_dict())
            for card, card_state in states.cards.items():
                self._write("card1_state", card, card_state.to_dict())

    def reset(self) -> None:
        self.load(HistoryStates())

    def row_count(self) -> int:
        with self._lock:
            return sum(
                self._db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                for t in ("uid_state", "card1_state")
            )
