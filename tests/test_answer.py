"""Chat-loop tests with a fake model and fake tools (no Ollama, no internet)."""
import asyncio
from types import SimpleNamespace as NS

from local_ai_assistant import client as chat

PYPI = "# fastmcp 4.0.11\n\nReleased: 2026-10-04"


def tool_msg(name, **args):
    return NS(content="", tool_calls=[NS(function=NS(name=name, arguments=args))])


def text_msg(text):
    return NS(content=text, tool_calls=None)


class FakeLLM:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.seen = []

    async def chat(self, **kwargs):
        self.seen.append(kwargs["messages"][-1])
        return NS(message=self.replies.pop(0))


class FakeClient:
    def __init__(self, text=PYPI):
        self.text, self.calls = text, []

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        return NS(content=[NS(type="text", text=self.text)])


def tools(*names):
    return [{"type": "function", "function": {"name": n, "description": "", "parameters": {}}} for n in names]


ALL = tools("pypi_info", "web_search", "fetch_url", "notes_search", "memory_save")


def run(llm, client, question="fastmcp ka latest version kya hai?"):
    messages = [{"role": "system", "content": "x"}, {"role": "user", "content": question}]
    asyncio.run(chat.answer(client, llm, ALL, messages, question))
    return messages


def test_good_answer_has_no_warning(capsys):
    llm = FakeLLM(tool_msg("pypi_info", package="fastmcp"), text_msg("FastMCP 4.0.11 hai, 2026-10-04 ko aaya."))
    client = FakeClient()
    run(llm, client)
    out = capsys.readouterr().out
    assert client.calls == [("pypi_info", {"package": "fastmcp"})]
    assert "4.0.11" in out and "⚠️" not in out


def test_invented_number_is_corrected(capsys):
    llm = FakeLLM(
        tool_msg("pypi_info", package="fastmcp"),
        text_msg("FastMCP 9.9.9 hai."),                     # lie
        text_msg("FastMCP 4.0.11 hai."),                    # fixed after correction
    )
    run(llm, FakeClient())
    out = capsys.readouterr().out
    assert "NOT in the tool result" in llm.seen[-1]["content"]
    assert "4.0.11" in out and "9.9.9" not in out and "⚠️" not in out


def test_persistent_lie_gets_warning(capsys):
    llm = FakeLLM(tool_msg("pypi_info", package="fastmcp"), text_msg("v 9.9.9"), text_msg("still 9.9.9"))
    run(llm, FakeClient())
    assert "9.9.9" in capsys.readouterr().out.split("⚠️")[1]


def test_answer_without_tool_is_nudged(capsys):
    llm = FakeLLM(
        text_msg("Version 1.2.3 hoga."),                    # guessed from memory
        tool_msg("pypi_info", package="fastmcp"),
        text_msg("FastMCP 4.0.11 hai."),
    )
    client = FakeClient()
    run(llm, client)
    assert client.calls and "without using a tool" in llm.seen[1]["content"]
    assert "⚠️" not in capsys.readouterr().out


def test_tool_call_written_as_text_is_executed():
    broken = text_msg('{"name":"notes_search","parameters{"query":"python"}}')
    llm = FakeLLM(broken, text_msg("Notes mein 1 match mila."))
    client = FakeClient("sub/b.txt:1: python")
    run(llm, client, question="meri notes mein python dhundo")
    assert client.calls == [("notes_search", {"query": "python"})]


def test_failed_tool_shows_real_error_not_generic_answer(capsys):
    class Failing(FakeClient):
        async def call_tool(self, name, args):
            raise RuntimeError("GitHub code search needs a token")

    llm = FakeLLM(tool_msg("pypi_info", package="x"))   # only ONE model reply is available
    run(llm, Failing(), question="github par code example dhundo")
    out = capsys.readouterr().out
    assert "Tool error" in out and "needs a token" in out and llm.replies == []
