"""Read-only access to ONE local folder (NOTES_DIR). Paths can never escape it."""
from __future__ import annotations

import os
from pathlib import Path

from .config import notes_dir

TEXT_EXT = {
    ".txt", ".md", ".markdown", ".rst", ".py", ".js", ".ts", ".json", ".csv",
    ".yaml", ".yml", ".toml", ".html", ".log", ".ini", ".sql",
}
MAX_FILE_BYTES = 2_000_000

WELCOME = (
    "# Notes folder\n\n"
    "Yeh aapka notes folder hai. Yahan .md / .txt files rakho (python seekhna hai, todo, diary...).\n"
    "Local Dev Assistant inhe sirf padh sakta hai, badal nahi sakta.\n"
)

def _default_in_use() -> bool:
    return not os.environ.get("NOTES_DIR")

def root() -> Path:
    r = notes_dir().expanduser()
    if not r.is_dir() and _default_in_use():
        r.mkdir(parents=True, exist_ok=True)
        (r / "welcome.md").write_text(WELCOME, encoding="utf-8")
    if not r.is_dir():
        raise ValueError(
            f"Notes folder not found: {r}. Create it, or fix NOTES_DIR in your .env file "
            "(or remove that line to use ~/notes)."
        )
    return r.resolve()

def status() -> tuple[str, str | None]:
    """(one-line description, warning or None) - shown when the chat starts."""
    try:
        base = root()
    except ValueError as exc:
        return str(exc), None
    count = len(list_files("", 10_000))
    warning = None
    if any((base / marker).exists() for marker in ("pyproject.toml", ".git", "src", "package.json")):
        warning = "NOTES_DIR project folder jaisa lagta hai. Apni notes ka alag folder set karo."
    return f"{base} ({count} files)", warning

def resolve(rel: str) -> Path:
    base = root()
    target = (base / rel).resolve()
    if target != base and not target.is_relative_to(base):
        raise ValueError("Path is outside the notes folder.")
    return target

def _visible(path: Path, base: Path) -> bool:
    return not any(part.startswith(".") for part in path.relative_to(base).parts)

def list_files(subdir: str = "", limit: int = 100) -> list[str]:
    base = root()
    folder = resolve(subdir)
    if not folder.is_dir():
        raise ValueError(f"Not a folder: {subdir!r}")
    out: list[str] = []
    for p in sorted(folder.rglob("*")):
        if p.is_file() and _visible(p, base) and p.suffix.lower() in TEXT_EXT | {".pdf"}:
            out.append(str(p.relative_to(base)))
            if len(out) >= limit:
                break
    return out

def read_file(path: str) -> str:
    target = resolve(path)
    if not target.is_file():
        raise ValueError(f"File not found: {path!r}")
    suffix = target.suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ModuleNotFoundError as exc:
            raise ValueError("PDF support needs: pip install 'local-ai-assistant[pdf]'") from exc
        reader = PdfReader(str(target))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    if suffix not in TEXT_EXT:
        raise ValueError(f"Unsupported file type: {suffix or '(none)'}")
    if target.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("File is too large (limit 2 MB).")
    return target.read_text(encoding="utf-8", errors="replace")

def search(query: str, limit: int = 10) -> list[str]:
    """Case-insensitive text search over all notes. Returns 'file:line: text' rows."""
    needle = query.strip().lower()
    if not needle:
        raise ValueError("Search text is empty.")
    base = root()
    hits: list[str] = []
    for p in sorted(base.rglob("*")):
        if not (p.is_file() and p.suffix.lower() in TEXT_EXT and _visible(p, base)):
            continue
        if p.stat().st_size > MAX_FILE_BYTES:
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), 1):
            if needle in line.lower():
                hits.append(f"[{p.relative_to(base)}] line {lineno}: {line.strip()[:200]}")
                if len(hits) >= limit:
                    return hits
    return hits

