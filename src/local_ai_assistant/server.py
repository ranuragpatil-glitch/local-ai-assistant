"""Local Dev Assistant server: research tools + local notes + long-term memory.

Run:  local-ai-assistant            (HTTP on 127.0.0.1:8000/mcp)
      local-ai-assistant --stdio    (for Claude Desktop etc.)
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import sys
from typing import Any, Awaitable, Callable
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import httpx
from bs4 import BeautifulSoup
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from markdownify import markdownify as html_to_md

from . import __version__, cache, memory, notes

log = logging.getLogger("devkit")


TIMEOUT = 30.0
MAX_FETCH_CHARS = 40_000
BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
API_UA = f"local-ai-assistant/{__version__}"

TTL_SEARCH = 60 * 60
TTL_PAGE = 60 * 60
TTL_PACKAGE = 6 * 60 * 60

mcp = FastMCP(
    name="Local Dev Assistant",
    instructions=(
        "Personal developer assistant. Pick the narrowest tool for the job.\n"
        "Research: web_search, search_docs, fetch_url, github_search_repos, "
        "github_search_code, github_get_file, stackoverflow_search, pypi_info.\n"
        "User's own files: notes_list, notes_read, notes_search.\n"
        "Long-term memory: memory_save, memory_search, memory_list, memory_delete.\n"
        "Prefer fetch_url on a result URL when a snippet is not enough."
    ),
)


def _client(*, browser: bool = False) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        headers={
            "User-Agent": BROWSER_UA if browser else API_UA,
            "Accept-Encoding": "gzip, deflate",
        },
        timeout=TIMEOUT,
        follow_redirects=True,
        transport=httpx.AsyncHTTPTransport(retries=2),
    )

async def _cached(ns: str, key: str, ttl: int, producer: Callable[[], Awaitable[str]]) -> str:
    """Return a cached string, or run `producer`, cache its result and return it."""
    full_key = f"{ns}:{key}"
    hit = cache.get(full_key)
    if hit is not None:
        log.info("cache hit: %s", full_key[:80])
        return hit
    value = await producer()
    cache.put(full_key, value, ttl)
    return value

def _truncate(text: str, limit: int = MAX_FETCH_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + f"\n\n… [truncated — {len(text) - limit} chars omitted]"

def _html_to_markdown(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(
        ["script", "style", "noscript", "nav", "footer", "header", "aside", "form", "iframe"]
    ):
        tag.decompose()
    md = html_to_md(str(soup), heading_style="ATX", bullets="-")
    return re.sub(r"\n{3,}", "\n\n", md).strip()

def _decode_ddg_url(href: str) -> str:
    """DuckDuckGo wraps outbound links in a `//duckduckgo.com/l/?uddg=…` redirect."""
    if href.startswith("//duckduckgo.com/l/") or href.startswith("/l/"):
        full = href if href.startswith("//") else "//duckduckgo.com" + href
        target = parse_qs(urlparse("https:" + full).query).get("uddg", [None])[0]
        if target:
            return unquote(target)
    return href

def _format_results(results: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for i, r in enumerate(results, 1):
        title = r.get("title") or "(untitled)"
        url = r.get("url") or ""
        snippet = (r.get("snippet") or "").strip()
        lines.append(f"{i}. **{title}**\n   {url}")
        if snippet:
            lines.append(f"   {snippet}")
    return "\n".join(lines)

def _limit(value: int, default: int, minimum: int = 500) -> int:
    """Models sometimes send max_chars=-1 or 0. Treat that as 'use the default'."""
    value = int(value)
    return default if value <= 0 else max(minimum, value)

def _clamp(n: int, low: int, high: int) -> int:
    return max(low, min(int(n), high))

def _github_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers

def _github_check(resp: httpx.Response) -> None:
    if resp.status_code in (401, 403):
        raise ToolError(
            "GitHub refused the request (rate limit or bad GITHUB_TOKEN). "
            "Check or set GITHUB_TOKEN in your .env file."
        )


async def _searxng_search(base_url: str, query: str, max_results: int) -> list[dict[str, Any]]:
    async with _client() as client:
        resp = await client.get(
            base_url.rstrip("/") + "/search", params={"q": query, "format": "json"}
        )
        resp.raise_for_status()
        data = resp.json()
    return [
        {"title": r.get("title"), "url": r.get("url"), "snippet": r.get("content")}
        for r in data.get("results", [])[:max_results]
    ]

async def _ddg_search(query: str, max_results: int) -> list[dict[str, Any]]:
    async with _client(browser=True) as client:
        resp = await client.post("https://html.duckduckgo.com/html/", data={"q": query})
        resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    results: list[dict[str, Any]] = []
    for block in soup.select("div.result")[:max_results]:
        anchor = block.select_one("a.result__a")
        if anchor is None:
            continue
        snippet_el = block.select_one(".result__snippet")
        results.append(
            {
                "title": anchor.get_text(" ", strip=True),
                "url": _decode_ddg_url(anchor.get("href", "")),
                "snippet": snippet_el.get_text(" ", strip=True) if snippet_el else "",
            }
        )
    return results

async def _search(query: str, max_results: int) -> list[dict[str, Any]]:
    max_results = _clamp(max_results, 1, 10)
    searx = os.environ.get("SEARXNG_URL", "").strip()
    if searx:
        try:
            results = await _searxng_search(searx, query, max_results)
            if results:
                return results
        except Exception as exc:  # noqa: BLE001 - fall back to DuckDuckGo
            log.warning("SearXNG failed (%s); falling back to DuckDuckGo", exc)
    return await _ddg_search(query, max_results)

async def _search_text(query: str, max_results: int, empty_msg: str) -> str:
    async def run() -> str:
        results = await _search(query, max_results)
        return _format_results(results) if results else empty_msg

    return await _cached("search", f"{query}|{max_results}", TTL_SEARCH, run)


@mcp.tool
async def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for general questions, blog posts and news. Returns titles, URLs, snippets."""
    return await _search_text(query, max_results, f"No results found for: {query}")

@mcp.tool
async def search_docs(query: str, site: str | None = None, max_results: int = 5) -> str:
    """Search official documentation. Pass `site` (e.g. 'docs.python.org') to limit to one site."""
    scoped = f"site:{site} {query}" if site else f"{query} documentation"
    return await _search_text(scoped, max_results, f"No documentation results for: {scoped}")

@mcp.tool
async def fetch_url(url: str, max_chars: int = MAX_FETCH_CHARS) -> str:
    """Download one web page (http/https) and return it as clean markdown text."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ToolError(f"Unsupported URL scheme: {parsed.scheme!r}")
    limit = _limit(max_chars, MAX_FETCH_CHARS, 1000)

    async def run() -> str:
        async with _client(browser=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
        ctype = resp.headers.get("content-type", "").lower()
        if "text/html" in ctype or "application/xhtml" in ctype:
            body = _html_to_markdown(resp.text)
        elif "application/json" in ctype:
            try:
                body = json.dumps(resp.json(), indent=2, ensure_ascii=False)
            except json.JSONDecodeError:
                body = resp.text
        elif ctype.startswith("text/") or "xml" in ctype:
            body = resp.text
        else:
            raise ToolError(
                f"Unsupported content type {ctype!r} at {url}. "
                "Only HTML, JSON, text and XML are supported."
            )
        return f"# Source: {url}\n\n" + _truncate(body, limit)

    return await _cached("page", f"{url}|{limit}", TTL_PAGE, run)

@mcp.tool
async def github_search_repos(
    query: str, language: str | None = None, sort: str = "stars", max_results: int = 5
) -> str:
    """Find GitHub repositories. sort = stars | forks | updated | best-match."""
    q = query if not language else f"{query} language:{language}"
    params: dict[str, Any] = {"q": q, "per_page": _clamp(max_results, 1, 20)}
    if sort in {"stars", "forks", "updated"}:
        params["sort"] = sort
    elif sort != "best-match":
        raise ToolError(f"Invalid sort: {sort!r}. Use stars, forks, updated, or best-match.")

    async def run() -> str:
        async with _client() as client:
            resp = await client.get(
                "https://api.github.com/search/repositories",
                params=params,
                headers=_github_headers(),
            )
            _github_check(resp)
            resp.raise_for_status()
            data = resp.json()
        items = data.get("items", [])
        if not items:
            return f"No repositories found for: {q}"
        lines = [f"GitHub repositories for **{q}** ({data.get('total_count', 0)} total):\n"]
        for i, repo in enumerate(items, 1):
            stars = repo.get("stargazers_count", 0)
            lang = repo.get("language") or "—"
            desc = (repo.get("description") or "").strip()
            lines.append(f"{i}. **{repo['full_name']}** — ⭐ {stars:,} · {lang}")
            lines.append(f"   {repo['html_url']}")
            if desc:
                lines.append(f"   {desc}")
            lines.append(f"   updated: {repo.get('updated_at', '?')[:10]}")
        return "\n".join(lines)

    return await _cached("ghrepos", json.dumps(params, sort_keys=True), TTL_SEARCH, run)

@mcp.tool
async def github_search_code(query: str, language: str | None = None, max_results: int = 5) -> str:
    """Find real code examples on GitHub. Needs GITHUB_TOKEN in .env."""
    if not os.environ.get("GITHUB_TOKEN", "").strip():
        raise ToolError(
            "GitHub code search needs a token. Add GITHUB_TOKEN=... to your .env file "
            "(GitHub > Settings > Developer settings > Personal access tokens) and restart."
        )
    q = query if not language else f"{query} language:{language}"
    async with _client() as client:
        resp = await client.get(
            "https://api.github.com/search/code",
            params={"q": q, "per_page": _clamp(max_results, 1, 20)},
            headers=_github_headers(),
        )
        _github_check(resp)
        resp.raise_for_status()
        data = resp.json()
    items = data.get("items", [])
    if not items:
        return f"No code matches for: {q}"
    lines = [f"Code matches for **{q}** ({data.get('total_count', 0)} total):\n"]
    for i, item in enumerate(items, 1):
        lines.append(f"{i}. `{item['repository']['full_name']}/{item['path']}`")
        lines.append(f"   {item['html_url']}")
    return "\n".join(lines)

@mcp.tool
async def github_get_file(
    owner: str, repo: str, path: str, ref: str | None = None, max_chars: int = MAX_FETCH_CHARS
) -> str:
    """Read one file (or list a folder) from a GitHub repo. `ref` = branch, tag or commit."""
    params = {"ref": ref} if ref else None
    async with _client() as client:
        resp = await client.get(
            f"https://api.github.com/repos/{quote_plus(owner)}/{quote_plus(repo)}/contents/{path.lstrip('/')}",
            params=params,
            headers=_github_headers(),
        )
        if resp.status_code == 404:
            raise ToolError(f"Not found: {owner}/{repo}/{path}" + (f"@{ref}" if ref else ""))
        _github_check(resp)
        resp.raise_for_status()
        data = resp.json()

    if isinstance(data, list):
        names = ", ".join(entry["name"] for entry in data[:50])
        return f"`{path}` is a directory. Contents:\n\n{names}"

    raw = data.get("content", "")
    if data.get("encoding") == "base64":
        try:
            text = base64.b64decode(raw).decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            raise ToolError(f"Failed to decode file: {exc}") from exc
    else:
        text = raw
    header = f"# {owner}/{repo} — {path} ({data.get('sha', '')[:7]})\n\n"
    return header + _truncate(text, _limit(max_chars, MAX_FETCH_CHARS, 1000))

@mcp.tool
async def stackoverflow_search(query: str, tagged: str | None = None, max_results: int = 5) -> str:
    """Search Stack Overflow questions for errors and how-to problems. `tagged` e.g. 'python'."""
    params: dict[str, Any] = {
        "order": "desc",
        "sort": "relevance",
        "q": query,
        "site": "stackoverflow",
        "pagesize": _clamp(max_results, 1, 15),
    }
    if tagged:
        params["tagged"] = tagged

    async def run() -> str:
        async with _client() as client:
            resp = await client.get("https://api.stackexchange.com/2.3/search/advanced", params=params)
            resp.raise_for_status()
            data = resp.json()
        items = data.get("items", [])
        if not items:
            return f"No Stack Overflow results for: {query}"
        lines = [f"Stack Overflow results for **{query}**:\n"]
        for i, item in enumerate(items, 1):
            answered = "✓ answered" if item.get("is_answered") else "· unanswered"
            lines.append(
                f"{i}. **{item['title']}** — score {item.get('score', 0)} · "
                f"{item.get('answer_count', 0)} answers · {answered}"
            )
            lines.append(f"   {item['link']}")
            if item.get("tags"):
                lines.append(f"   tags: {', '.join(item['tags'])}")
        return "\n".join(lines)

    return await _cached("so", json.dumps(params, sort_keys=True), TTL_SEARCH, run)

@mcp.tool
async def pypi_info(package: str) -> str:
    """Get a Python package's latest version, release date, license and dependencies from PyPI."""

    async def run() -> str:
        async with _client() as client:
            resp = await client.get(f"https://pypi.org/pypi/{quote_plus(package)}/json")
            if resp.status_code == 404:
                raise ToolError(f"Package not found on PyPI: {package}")
            resp.raise_for_status()
            data = resp.json()
        info = data.get("info", {})
        version = info.get("version", "?")
        files = data.get("urls") or []
        uploaded = files[0].get("upload_time_iso_8601", "?")[:10] if files else "?"
        lines = [
            f"# {info.get('name', package)} {version}",
            "",
            f"**Summary:** {info.get('summary') or '—'}",
            f"**Released:** {uploaded}",
            f"**License:** {info.get('license_expression') or info.get('license') or '—'}",
            f"**Requires Python:** {info.get('requires_python') or '—'}",
        ]
        if info.get("home_page"):
            lines.append(f"**Homepage:** {info['home_page']}")
        project_urls = info.get("project_urls") or {}
        for label in ("Source", "Repository", "Documentation", "Changelog"):
            if label in project_urls:
                lines.append(f"**{label}:** {project_urls[label]}")
        deps = info.get("requires_dist") or []
        if deps:
            lines += ["", "**Dependencies:**"]
            lines += [f"- {dep}" for dep in deps[:25]]
            if len(deps) > 25:
                lines.append(f"- … and {len(deps) - 25} more")
        return "\n".join(lines)

    return await _cached("pypi", package.lower(), TTL_PACKAGE, run)


@mcp.tool
def notes_list(subdir: str = "", limit: int = 100) -> str:
    """List the user's own notes/files (txt, md, code, pdf). `subdir` is relative to the notes folder."""
    try:
        files = notes.list_files(subdir, _clamp(limit, 1, 500))
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    return "\n".join(files) if files else "No files found."

@mcp.tool
def notes_read(path: str, max_chars: int = 20_000) -> str:
    """Read one of the user's own files. `path` is relative to the notes folder (see notes_list)."""
    try:
        text = notes.read_file(path)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    return _truncate(text, _limit(max_chars, 20_000))

@mcp.tool
def notes_search(query: str, max_results: int = 10) -> str:
    """Search inside the user's own notes for a word or phrase. Returns file:line: text."""
    try:
        hits = notes.search(query, _clamp(max_results, 1, 50))
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    return "\n".join(hits) if hits else f"No matches in notes for: {query}"


def _format_memories(rows: list[tuple]) -> str:
    if not rows:
        return "No memories found."
    lines = []
    for mid, text, tags, created in rows:
        tag_part = f" [tags: {tags}]" if tags else ""
        lines.append(f"#{mid} ({created}){tag_part}: {text}")
    return "\n".join(lines)

@mcp.tool
def memory_save(text: str, tags: str = "") -> str:
    """Save one short fact or note to long-term memory (e.g. user preferences, decisions). Optional comma-separated tags."""
    try:
        mid = memory.save(text, tags)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    return f"Saved memory #{mid}."

@mcp.tool
def memory_search(query: str, limit: int = 10) -> str:
    """Search saved memories. Every word in `query` must match the text or tags."""
    return _format_memories(memory.search(query, limit))

@mcp.tool
def memory_list(limit: int = 10) -> str:
    """Show the most recent saved memories."""
    return _format_memories(memory.recent(limit))

@mcp.tool
def memory_delete(memory_id: int) -> str:
    """Delete one saved memory by its number (the #id shown in memory_list)."""
    return f"Deleted memory #{memory_id}." if memory.delete(memory_id) else f"No memory #{memory_id}."


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(message)s")
    if "--stdio" in sys.argv:
        mcp.run()
    else:
        host = os.environ.get("DEVKIT_HOST", "127.0.0.1")
        port = int(os.environ.get("DEVKIT_PORT", "8000"))
        mcp.run(transport="http", host=host, port=port)

if __name__ == "__main__":
    main()

