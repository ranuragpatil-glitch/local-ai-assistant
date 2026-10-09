"""Safety nets for small local models (pure functions, easy to test).

1. clean_args / keep_schema_args  - repair bad tool arguments
2. salvage_tool_call              - model WROTE a tool call as text instead of calling it
3. select_tools                   - give the model only the tools that fit the question
4. ungrounded_tokens              - catch numbers/links in the answer that no tool returned
5. to_roman                       - Hindi script -> English letters (terminal friendly)
"""
from __future__ import annotations

import json
import re


def clean_args(args: dict | None) -> dict:
    """Drop None / "null" / "" values that small models often send by mistake."""
    cleaned = {}
    for key, value in (args or {}).items():
        if value is None:
            continue
        if isinstance(value, str) and value.strip().lower() in ("null", "none", ""):
            continue
        cleaned[key] = value
    return cleaned

def keep_schema_args(args: dict, schema: dict | None) -> dict:
    """Remove arguments the tool does not accept (small models invent extra ones)."""
    allowed = set(((schema or {}).get("properties") or {}).keys())
    return {k: v for k, v in args.items() if not allowed or k in allowed}


def salvage_tool_call(content: str | None, names: set[str]) -> tuple[str, dict] | None:
    """Recover tool name + arguments from text like {"name": "...", "parameters": {...}}
    (even when the JSON is broken)."""
    text = (content or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    match = re.search(r'"name"\s*:\s*"(\w+)"', text)
    if not match or match.group(1) not in names:
        return None
    name = match.group(1)

    fixed = re.sub(r'"(parameters|arguments)"?\s*:?\s*\{', r'"\1":{', text)
    args: dict = {}
    try:
        obj = json.loads(fixed)
        if isinstance(obj, list) and obj:
            obj = obj[0]
        if isinstance(obj, dict):
            args = obj.get("parameters") or obj.get("arguments") or {}
    except json.JSONDecodeError:
        for key, value in re.findall(r'"(\w+)"\s*:\s*("(?:[^"\\]|\\.)*"|-?\d+)', text):
            if key not in ("name", "parameters", "arguments"):
                args[key] = json.loads(value)
    return name, args if isinstance(args, dict) else {}


TOOL_GROUPS: dict[str, tuple[set[str], list[str]]] = {
    "notes": (
        {"notes_list", "notes_read", "notes_search"},
        ["note", "meri file", "meri files", "diary", "todo", "folder"],
    ),
    "memory": (
        {"memory_save", "memory_search", "memory_list", "memory_delete"},
        ["yaad", "remember", "memory", "bhool", "forget", "mujhe kya pasand", "meri pasand"],
    ),
    "package": (
        {"pypi_info"},
        ["version", "package", "pypi", "pip ", "library", "license", "dependenc"],
    ),
    "github": (
        {"github_search_repos", "github_search_code", "github_get_file"},
        ["github", "repo", "repository"],
    ),
    "stack": (
        {"stackoverflow_search"},
        ["error", "exception", "traceback", "stackoverflow", "stack overflow", "bug"],
    ),
    "docs": ({"search_docs"}, ["docs", "documentation"]),
}
WEB_TOOLS = {"web_search", "search_docs", "fetch_url", "stackoverflow_search"}
MUST_USE_TOOL = {"notes", "memory", "package", "github"}

def match_groups(question: str) -> list[str]:
    q = question.lower()
    return [g for g, (_, words) in TOOL_GROUPS.items() if any(w in q for w in words)]

def select_tools(question: str, tools: list[dict]) -> tuple[list[dict], list[str]]:
    """Pick the tools that fit the question. Falls back to all tools if unsure."""
    groups = match_groups(question)
    wanted: set[str] = set()
    for g in groups:
        wanted |= TOOL_GROUPS[g][0]
    if groups and not ({"notes", "memory"} & set(groups)):
        wanted |= {"web_search", "fetch_url"}
    has_url = bool(re.search(r"https?://", question))
    if has_url:
        wanted |= {"fetch_url", "web_search"}
    if not groups and not has_url:
        if re.search(r"\b(search|dhundo|google|news|latest)\b", question.lower()):
            wanted |= WEB_TOOLS
        else:
            return tools, groups
    chosen = [t for t in tools if t["function"]["name"] in wanted]
    return (chosen or tools), groups

def requires_tool(groups: list[str]) -> bool:
    return bool(MUST_USE_TOOL & set(groups))


_TOKEN = re.compile(r"https?://[^\s)>\]]+|\d+(?:[.\-:/]\d+)+|\d{3,}")

def ungrounded_tokens(answer: str, sources: list[str], question: str = "") -> list[str]:
    """Versions, dates, big numbers and links that appear in `answer` but in NO tool result."""
    haystack = "\n".join(sources) + "\n" + question
    bad: list[str] = []
    for tok in _TOKEN.findall(answer or ""):
        tok = tok.rstrip(".,;:)")
        if tok and tok not in haystack and tok not in bad:
            bad.append(tok)
    return bad


DEVANAGARI = re.compile(r"[\u0900-\u097F]")
_DEV_RUN = re.compile(r"[\u0900-\u097F]+(?:[ \t]+[\u0900-\u097F]+)*")
_warned_missing_lib = False

def _hinglishify(itrans: str) -> str:
    """ITRANS uses capitals for long vowels (kA, yAda). Make it look like Hinglish."""
    for old, new in (("A", "aa"), ("I", "ee"), ("U", "oo"), ("M", "n"), ("|", ".")):
        itrans = itrans.replace(old, new)
    return itrans.lower()

def to_roman(text: str) -> str:
    """If the model still answers in Devanagari, convert it to English letters (offline)."""
    global _warned_missing_lib
    if not text or not DEVANAGARI.search(text):
        return text
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate
    except ImportError:
        if not _warned_missing_lib:
            print("ℹ️  Hindi script ko English letters mein badalne ke liye: uv add indic-transliteration")
            _warned_missing_lib = True
        return text
    return _DEV_RUN.sub(
        lambda m: _hinglishify(transliterate(m.group(0), sanscript.DEVANAGARI, sanscript.ITRANS)),
        text,
    )


def trim_history(messages: list, max_history: int) -> list:
    """Keep the system prompt + the last `max_history` messages, starting at a user turn."""
    if len(messages) <= max_history + 1:
        return messages
    tail = messages[-max_history:]

    def role(m):
        return m.get("role") if isinstance(m, dict) else getattr(m, "role", "")

    while tail and role(tail[0]) != "user":
        tail = tail[1:]
    return [messages[0]] + tail

def format_notes_result(content: str) -> str:
    if not content or content.startswith("No matches") or content == "No files found.":
        return content
    lines = content.strip().splitlines()
    return f"NOTES RESULTS ({len(lines)} lines):\n" + "\n".join(f"  • {line}" for line in lines)

