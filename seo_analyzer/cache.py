"""Дисковый кэш на SQLite: HTML, выдача, бэклинки, эмбеддинги.

Повторный запуск по тем же ключам не тратит деньги на API.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np


class Cache:
    def __init__(self, path: Path | str | None):
        self._lock = threading.Lock()
        self.enabled = path is not None
        if not self.enabled:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS kv (ns TEXT, key TEXT, ts REAL, value BLOB, PRIMARY KEY (ns, key))"
        )
        self._db.commit()

    @staticmethod
    def _k(key: Any) -> str:
        raw = key if isinstance(key, str) else json.dumps(key, sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def get_json(self, ns: str, key: Any, ttl_hours: float | None = None) -> Any | None:
        row = self._get(ns, key, ttl_hours)
        return None if row is None else json.loads(row.decode("utf-8"))

    def set_json(self, ns: str, key: Any, value: Any) -> None:
        self._set(ns, key, json.dumps(value, ensure_ascii=False).encode("utf-8"))

    def get_vec(self, ns: str, key: Any) -> np.ndarray | None:
        row = self._get(ns, key, None)
        return None if row is None else np.frombuffer(row, dtype=np.float32).copy()

    def set_vec(self, ns: str, key: Any, vec: np.ndarray) -> None:
        self._set(ns, key, np.asarray(vec, dtype=np.float32).tobytes())

    def _get(self, ns: str, key: Any, ttl_hours: float | None) -> bytes | None:
        if not self.enabled:
            return None
        with self._lock:
            cur = self._db.execute("SELECT ts, value FROM kv WHERE ns=? AND key=?", (ns, self._k(key)))
            row = cur.fetchone()
        if row is None:
            return None
        ts, value = row
        if ttl_hours is not None and time.time() - ts > ttl_hours * 3600:
            return None
        return value

    def _set(self, ns: str, key: Any, value: bytes) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO kv (ns, key, ts, value) VALUES (?, ?, ?, ?)",
                (ns, self._k(key), time.time(), value),
            )
            self._db.commit()
