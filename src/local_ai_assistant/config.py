"""Small helpers for paths and settings (all read from environment / .env)."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

def data_dir() -> Path:
    """Folder where cache.db and memory.db live. Default: ~/.local-ai-assistant"""
    path = Path(os.environ.get("DEVKIT_HOME") or Path.home() / ".local-ai-assistant").expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path

def notes_dir() -> Path:
    """Folder the local-file tools are allowed to read. Default: ~/notes"""
    return Path(os.environ.get("NOTES_DIR") or Path.home() / "notes").expanduser()
