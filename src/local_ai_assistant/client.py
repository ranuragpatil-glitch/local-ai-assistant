"""Terminal chat: local Ollama model + Local Dev Assistant tools.

Just run `local-ai-chat` - the tools start inside the same program (no second terminal).
To use a separately running server instead, set DEVKIT_URL=http://127.0.0.1:8000/mcp
"""
from __future__ import annotations

import asyncio
import os

import ollama
from fastmcp import Client

from . import config  # noqa: F401  (loads .env)
from . import notes
from .guards import (
    clean_args,
    format_notes_result,
    keep_schema_args,
    requires_tool,
    salvage_tool_call,
    select_tools,
    to_roman,
    trim_history,
    ungrounded_tokens,
)

MODEL = os.environ.get("OLLAMA_MODEL") or "qwen2.5:7b"
NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX") or "8192")
TEMPERATURE = float(os.environ.get("OLLAMA_TEMPERATURE") or "0.2")
SERVER_URL = os.environ.get("DEVKIT_URL", "").strip()
MAX_TOOL_ROUNDS = 6
MAX_HISTORY = 24
RESULT_LIMIT = 6000

LANG_HINT = "\n\n[Reply in English letters only: simple English or Roman Hinglish. No Devanagari script.]"
GROUND_PROMPT = (
    "Use ONLY the tool result above to answer. Do not add facts, numbers, versions or links "
    "that are not in it. If the answer is not there, say: Tool se ye jaankari nahi mili. "
    "Reply in English letters only."
)
NUDGE_PROMPT = (
    "You answered without using a tool. Do NOT answer from memory. "
    "Call the correct tool now and answer only from its result."
)

SYSTEM_PROMPT = """You are a clean and precise personal developer assistant.

LANGUAGE: Reply in simple English, or Hinglish written with English letters (a-z) only.
Example: "Aapka sawal bahut accha hai". NEVER use Devanagari/Hindi script.

STRICT RULES:
1. Always call the correct tool. Do not guess facts you can look up.
2. Package version / dependencies / license -> ONLY use `pypi_info`.
3. GitHub repositories -> use `github_search_repos` with sort="stars".
4. User's own notes/files -> use `notes_search`, `notes_read`, or `notes_list`.
5. "yaad rakhna" / "remember" / preference -> immediately call `memory_save`.
   Write the fact in third person, e.g. text="User likes chai" (not "I like chai").
6. Personal questions ("mujhe kya pasand hai?") -> call `memory_search` first.
7. Never send "null", empty string, or None as argument values.
8. NEVER invent information. Only use what the tool actually returned.
9. In notes results the number after a file name is a LINE NUMBER, not a version.
   If the tool says "No matches" -> reply exactly: "Notes mein yeh nahi mila."
10. Keep answers short and clean.
"""

def mcp_tools_to_ollama(tools) -> list[dict]:
    out = []
    for tool in tools:
        schema = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None) or {}
        out.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": schema,
                },
            }
        )
    return out

def result_text(result) -> str:
    text = "".join(
        block.text
        for block in (getattr(result, "content", None) or [])
        if getattr(block, "type", None) == "text"
    )
    return text or str(getattr(result, "data", result))

def make_client() -> tuple[Client, str]:
    """Built-in tools by default; an external server only if DEVKIT_URL is set."""
    if SERVER_URL:
        return Client(SERVER_URL), f"server: {SERVER_URL}"
    from .server import mcp

    return Client(mcp), "built-in tools"

def _box(title: str, body: str) -> None:
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)
    print(body)
    print("=" * 50)

async def answer(
    client: Client, llm: ollama.AsyncClient, all_tools: list[dict], messages: list, question: str
) -> None:
    tools, groups = select_tools(question, all_tools)
    need_tool = requires_tool(groups)
    names = {t["function"]["name"] for t in all_tools}
    schemas = {t["function"]["name"]: t["function"]["parameters"] for t in all_tools}
    sources: list[str] = []
    nudged = corrected = False
    print(f"  ({len(tools)} tools model ko diye gaye)")

    for _ in range(MAX_TOOL_ROUNDS):
        response = await llm.chat(
            model=MODEL,
            messages=messages,
            tools=tools,
            options={"num_ctx": NUM_CTX, "temperature": TEMPERATURE},
        )
        msg = response.message
        messages.append(msg)

        calls: list[tuple[str, dict]] = []
        if msg.tool_calls:
            calls = [(c.function.name, clean_args(c.function.arguments)) for c in msg.tool_calls]
        else:
            recovered = salvage_tool_call(msg.content, names)
            if recovered:
                name, args = recovered
                messages.pop()
                messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"function": {"name": name, "arguments": clean_args(args)}}],
                    }
                )
                calls = [(name, clean_args(args))]

        if not calls:
            text = msg.content or ""
            if need_tool and not sources and not nudged:
                nudged = True
                messages.append({"role": "user", "content": NUDGE_PROMPT})
                continue
            bad = ungrounded_tokens(text, sources, question) if sources else []
            if bad and not corrected:
                corrected = True
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your answer mentions {', '.join(bad)} which is NOT in the tool result. "
                            "Rewrite using ONLY the tool result. If it is missing, say: "
                            "Tool se ye jaankari nahi mili."
                        ),
                    }
                )
                continue
            final = to_roman(text)
            msg.content = final
            _box("🟢 Answer:", final)
            if need_tool and not sources:
                print("⚠️  Model ne koi tool use nahi kiya, isliye ye jawab pakka nahi hai.")
            elif bad:
                print(f"⚠️  Ye cheezein tool data mein nahi mili, verify karo: {', '.join(bad)}")
            return

        errors: list[str] = []
        for name, args in calls:
            args = keep_schema_args(args, schemas.get(name))
            print(f"\n→ Tool: {name}\n  Args: {args}")
            try:
                result = await client.call_tool(name, args)
                content = result_text(result)[:RESULT_LIMIT]
                print("← Tool se data mil gaya")
            except Exception as exc:  # noqa: BLE001
                content = f"Tool failed: {exc}"
                errors.append(str(exc))
                print(f"❌ {content}")
            if name.startswith("notes_"):
                content = format_notes_result(content)
            if name == "memory_save" and "Saved memory" in content:
                _box("🟢 Answer:", "Memory save ho gayi. " + content)
                return
            sources.append(content)
            messages.append({"role": "tool", "tool_name": name, "content": content})

        if errors and len(errors) == len(calls):
            _box("⚠️ Tool error:", "\n".join(errors))
            return
        called_names = {name for name, _ in calls}
        q_lower = question.lower()
        if (called_names == {"web_search"} and
                any(w in q_lower for w in ["page bhi padho", "link padho", "url padho", "open karo", "fetch karo", "padh bhi"])):
            messages.append({
                "role": "user",
                "content": "Ab top result ka URL fetch_url se padho aur uska content bhi include karo jawab mein."
            })
        else:
            messages.append({"role": "user", "content": GROUND_PROMPT})

    print("\n⚠️ Bahut zyada tool calls ho gaye. Sawal thoda simple karke dobara pucho.")

async def check_model(llm: ollama.AsyncClient) -> bool:
    """Make sure the model is downloaded; otherwise tell the user what to run."""
    try:
        installed = {m.model for m in (await llm.list()).models}
    except Exception as exc:  # noqa: BLE001
        print(f"❌ Ollama se baat nahi ho pa rahi ({exc}). Ollama app chalu hai?")
        return False
    wanted = MODEL if ":" in MODEL else MODEL + ":latest"
    if wanted not in installed:
        print(f"❌ Model '{MODEL}' download nahi hai. Chalao: ollama pull {MODEL}")
        print("   Downloaded models:", ", ".join(sorted(installed)) or "koi nahi")
        return False
    return True

async def run() -> None:
    llm = ollama.AsyncClient()
    if not await check_model(llm):
        return
    client, where = make_client()
    try:
        async with client:
            tools = mcp_tools_to_ollama(await client.list_tools())
            notes_line, notes_warning = notes.status()
            print(f"✅ Ready | model: {MODEL} | {where} | {len(tools)} tools")
            print(f"📁 Notes: {notes_line}")
            if notes_warning:
                print(f"⚠️  {notes_warning}")
            print("-" * 55)

            messages: list = [{"role": "system", "content": SYSTEM_PROMPT}]
            while True:
                try:
                    user_input = input("\nAapka sawal (quit = band): ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if user_input.lower() in ("quit", "exit", "q"):
                    break
                if not user_input:
                    continue
                messages.append({"role": "user", "content": user_input + LANG_HINT})
                try:
                    await answer(client, llm, tools, messages, user_input)
                except Exception as exc:  # noqa: BLE001
                    print(f"❌ Error: {exc}")
                messages[:] = trim_history(messages, MAX_HISTORY)
    except Exception as exc:  # noqa: BLE001
        if SERVER_URL:
            print(f"❌ Server ({SERVER_URL}) se connect nahi hua: {exc}")
            print("   Pehle `uv run local-ai-assistant` chalao, ya DEVKIT_URL hata do.")
        else:
            raise

def main() -> None:
    asyncio.run(run())

if __name__ == "__main__":
    main()

