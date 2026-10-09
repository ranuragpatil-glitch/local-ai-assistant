from local_ai_assistant.guards import (
    clean_args,
    keep_schema_args,
    requires_tool,
    salvage_tool_call,
    select_tools,
    to_roman,
    trim_history,
    ungrounded_tokens,
)

NAMES = {"notes_search", "memory_save", "pypi_info"}


def fake_tools(*names):
    return [{"type": "function", "function": {"name": n, "description": "", "parameters": {}}} for n in names]


ALL = fake_tools(
    "web_search", "search_docs", "fetch_url", "github_search_repos", "github_search_code",
    "github_get_file", "stackoverflow_search", "pypi_info", "notes_list", "notes_read",
    "notes_search", "memory_save", "memory_search", "memory_list", "memory_delete",
)


def names(tools):
    return {t["function"]["name"] for t in tools}


def test_clean_args_drops_null_like_values():
    assert clean_args({"a": None, "b": "null", "c": " ", "d": "x", "e": 0}) == {"d": "x", "e": 0}


def test_keep_schema_args():
    assert keep_schema_args({"q": 1, "junk": 2}, {"properties": {"q": {}}}) == {"q": 1}
    assert keep_schema_args({"q": 1}, None) == {"q": 1}


def test_salvage_broken_json_from_screenshot():
    text = '{"name":"notes_search","parameters{"max_results":10,"query":"python"}}'
    assert salvage_tool_call(text, NAMES) == ("notes_search", {"max_results": 10, "query": "python"})


def test_salvage_fenced_and_unknown():
    fenced = '```json\n{"name":"pypi_info","arguments":{"package":"fastmcp"}}\n```'
    assert salvage_tool_call(fenced, NAMES) == ("pypi_info", {"package": "fastmcp"})
    assert salvage_tool_call("normal answer", NAMES) is None
    assert salvage_tool_call('{"name":"nope","parameters":{}}', NAMES) is None


def test_routing_package_question():
    chosen, groups = select_tools("fastmcp ka latest version kya hai?", ALL)
    assert "pypi_info" in names(chosen) and "notes_search" not in names(chosen)
    assert requires_tool(groups)


def test_routing_notes_and_memory_exclude_web():
    chosen, _ = select_tools('meri notes mein "python" dhundo', ALL)
    assert names(chosen) == {"notes_list", "notes_read", "notes_search"}
    chosen, groups = select_tools("yaad rakhna mujhe chai pasand hai", ALL)
    assert "memory_save" in names(chosen) and "web_search" not in names(chosen)
    assert requires_tool(groups)


def test_routing_unknown_question_gets_everything():
    chosen, groups = select_tools("kya haal hai", ALL)
    assert len(chosen) == len(ALL) and not requires_tool(groups)


def test_ungrounded_tokens():
    src = ["# fastmcp 4.0.11\nReleased: 2026-10-04\nhttps://pypi.org/project/fastmcp/"]
    assert ungrounded_tokens("FastMCP 4.0.11, 2026-10-04.", src) == []
    assert ungrounded_tokens("FastMCP 9.9.9 aur https://evil.example", src) == [
        "9.9.9", "https://evil.example",
    ]
    assert ungrounded_tokens("Aap ne 2026 poocha", [], "2026 kab?") == []


def test_to_roman_converts_hindi_and_keeps_english():
    out = to_roman("FastMCP का वर्शन 4.0.11 है।")
    assert "FastMCP" in out and "4.0.11" in out
    assert not any("\u0900" <= ch <= "\u097f" for ch in out)
    assert to_roman("plain english") == "plain english"


def test_trim_history_starts_with_user():
    msgs = [{"role": "system"}] + [
        {"role": r} for r in ["user", "assistant", "tool", "user", "assistant"] * 10
    ]
    out = trim_history(msgs, 7)
    assert out[0]["role"] == "system" and out[1]["role"] == "user" and len(out) <= 8


def test_routing_url_question_gets_only_fetch_tools():
    chosen, _ = select_tools("https://example.com padho aur batao isme kya likha hai", ALL)
    assert names(chosen) == {"fetch_url", "web_search"}
