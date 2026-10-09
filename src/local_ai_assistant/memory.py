"""Long-term memory in SQLite: save short facts and search them later."""
from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timezone

from .config import data_dir

def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(data_dir() / "memory.db")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS memories ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL, "
        "tags TEXT NOT NULL DEFAULT '', created TEXT NOT NULL)"
    )
    return conn

def save(text: str, tags: str = "") -> int:
    text = text.strip()
    if not text:
        raise ValueError("Memory text is empty.")
    created = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    with closing(_conn()) as conn, conn:
        cur = conn.execute(
            "INSERT INTO memories (text, tags, created) VALUES (?, ?, ?)",
            (text, tags.strip(), created),
        )
        return int(cur.lastrowid)

def search(query: str, limit: int = 10) -> list[tuple]:
    """Every word in `query` must appear in the text or tags (case-insensitive)."""
    words = query.split()
    if not words:
        return recent(limit)
    where = " AND ".join("(text LIKE ? OR tags LIKE ?)" for _ in words)
    params: list = []
    for word in words:
        like = f"%{word}%"
        params += [like, like]
    params.append(max(1, min(int(limit), 50)))
    with closing(_conn()) as conn:
        return conn.execute(
            f"SELECT id, text, tags, created FROM memories WHERE {where} "
            "ORDER BY id DESC LIMIT ?",
            params,
        ).fetchall()

def recent(limit: int = 10) -> list[tuple]:
    with closing(_conn()) as conn:
        return conn.execute(
            "SELECT id, text, tags, created FROM memories ORDER BY id DESC LIMIT ?",
            (max(1, min(int(limit), 50)),),
        ).fetchall()

def delete(memory_id: int) -> bool:
    with closing(_conn()) as conn, conn:
        return conn.execute("DELETE FROM memories WHERE id = ?", (int(memory_id),)).rowcount > 0

