import asyncio

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from local_ai_assistant.server import _limit, mcp


def test_limit_ignores_silly_values():
    assert _limit(-1, 40_000, 1000) == 40_000
    assert _limit(0, 40_000, 1000) == 40_000
    assert _limit(10, 40_000, 1000) == 1000
    assert _limit(5000, 40_000, 1000) == 5000


def test_code_search_without_token_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    async def go():
        async with Client(mcp) as c:
            await c.call_tool("github_search_code", {"query": "x"})

    with pytest.raises(ToolError, match="GITHUB_TOKEN"):
        asyncio.run(go())
