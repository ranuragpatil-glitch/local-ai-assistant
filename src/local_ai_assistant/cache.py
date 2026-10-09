"""Tiny SQLite cache with expiry (TTL). Set DEVKIT_CACHE=0 to turn it off."""
from __future__ import annotations

import os
import sqlite3
import time
from contextlib import closing

from .config import data_dir

def _enabled() -> bool:
    return os.environ.get("DEVKIT_CACHE", "1") != "0"

def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(data_dir() / "cache.db")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS cache "
        "(key TEXT PRIMARY KEY, value TEXT NOT NULL, expires REAL NOT NULL)"
    )
    return conn

def get(key: str) -> str | None:
    if not _enabled():
        return None
    with closing(_conn()) as conn:
        row = conn.execute("SELECT value, expires FROM cache WHERE key = ?", (key,)).fetchone()
    if row and row[1] > time.time():
        return row[0]
    return None

def put(key: str, value: str, ttl: int) -> None:
    if not _enabled():
        return
    now = time.time()
    with closing(_conn()) as conn, conn:
        conn.execute("DELETE FROM cache WHERE expires < ?", (now,))
        conn.execute(
            "INSERT OR REPLACE INTO cache (key, value, expires) VALUES (?, ?, ?)",
            (key, value, now + ttl),
        )

def clear() -> int:
    with closing(_conn()) as conn, conn:
        return conn.execute("DELETE FROM cache").rowcount

