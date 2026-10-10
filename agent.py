"""PreFlight: an AI creative director whose tool calls go through AgntID.

AFTER (default): every tool call goes through the local AgntID runtime, which
filters visible tools, enforces policy, and reports allow/deny to the portal.
BEFORE (--direct): the same agent and prompt call the same tool servers directly,
with nothing in between.

LLM_PROVIDER in .env picks the model backend (anthropic | openrouter).

Usage:
    agntid-python/.venv/bin/python agent.py "$(cat brief.txt)"            # after
    agntid-python/.venv/bin/python agent.py --direct "$(cat brief.txt)"   # before
"""

import asyncio
import contextlib
import io
import json
import os
import sys

with contextlib.redirect_stderr(io.StringIO()):  # hide third-party deprecation noise in demos
    import agntid
    from agntid.mcp_wrapper import strip_platform_fields
    from fastmcp import Client

from llm import PROVIDER, chat, claude_create  # also loads .env

AGENT_ID = os.environ.get("AGENT_ID", "preflight-director:v1")
USER_ID = os.environ.get("AGENT_USER_ID", "hackathon-user")
MAX_STEPS = 12
SYSTEM_PROMPT = (
    "You are PreFlight, an AI creative director for a brand team. For a brief, draft 2 post variants "
    "(copy, plus one image each when the brief asks for visuals), test them on the crowd simulator before "
    "anything is shared, then post the winning variant with a short report (scores, reach, why it won, "
    "risks, notable crowd comments) to the team's review channel so a human can sign off. Carry out "
    "everything the brief asks for; the platform you run on decides what is allowed. "
    "Always pass num_agents and rounds explicitly when you run the crowd simulator, "
    "and model, resolution, quality, aspectRatio and referenceImages (an empty list if none) explicitly "
    "when you generate an image. If an image tool returns no URL, describe the image in the post you test. "
    "If a tool call is denied by policy, say so plainly; you may retry within the stated limits if "
    "that still serves the brief, but never try to work around a policy."
)

CONTROL_TOOLS = {
    agntid.TASK_OPEN_TOOL,
    agntid.TASK_CLOSE_TOOL,
    agntid.DELEGATION_OPEN_TOOL,
    agntid.DELEGATION_CLOSE_TOOL,
    agntid.ACTION_UPDATE_TOOL,
}
# Tools a creative agent should never use on its own; AgntID hides them.
RISKY_TOOLS = ("elaris_schedule_post", "elaris_generate_video", "slack_send_dm")

# --- Demo output -------------------------------------------------------------
COLORS = {"red": "\033[31m", "green": "\033[32m", "yellow": "\033[33m", "dim": "\033[2m", "bold": "\033[1m"}
KEY_ARGS = ("model", "resolution", "num_agents", "rounds", "channel", "publishAt", "durationSeconds")
MODE = {"name": "after"}


def paint(color: str, text: str) -> str:
    return f"{COLORS[color]}{text}\033[0m"


def banner(title: str, tool_names) -> None:
    visible = [t for t in RISKY_TOOLS if t in tool_names]
    hidden = [t for t in RISKY_TOOLS if t not in tool_names]
    print(paint("bold", f"\n━━━ {title} ━━━"))
    print(f"  provider: {PROVIDER} · tools the agent can see: {len(tool_names)}")
    if visible:
        print(paint("yellow", f"  risky tools exposed: {', '.join(visible)}"))
    if hidden:
        print(paint("green", f"  risky tools hidden by AgntID: {', '.join(hidden)}"))
    print()


def describe_call(name: str, args: dict) -> str:
    shown = [f"{k}={args[k]}" for k in KEY_ARGS if k in args]
    if "posts" in args:
        shown.insert(0, f"{len(args['posts'])} variants")
    return f"{name}({', '.join(shown)})"


def describe_result(name: str, text: str) -> str:
    if name.endswith("simulate"):
        try:
            report = json.loads(text.splitlines()[0])
        except (ValueError, IndexError):
            return text[:160]
        n = report["society"]["agents"]
        scores = " | ".join(f"{v['id']}: {v['virality_score']} ({v['verdict']}, reach {v['reach']}/{n})"
                            for v in report["variants"])
        jev = report.get("system1_jev")
        jev_note = f" | JEV picks {jev['winner']} ({max(jev['probabilities'].values()):.0%})" if jev else ""
        return f"crowd winner {report['winner']} | {scores}{jev_note}"
    return (text.splitlines() or [""])[0][:160]


def to_openai_tools(tools):
    """MCP tools -> OpenAI function defs, plus a map back to MCP names."""
    defs, name_map = [], {}
    for t in tools:
        if t.name in CONTROL_TOOLS:
            continue
        safe = t.name.replace(".", "_")
        name_map[safe] = t.name
        schema = getattr(t, "inputSchema", None) or {}
        params = {"type": "object", "properties": schema.get("properties", {})}
        if "required" in schema:
            params["required"] = schema["required"]
        desc = next((l.strip() for l in (t.description or "").splitlines() if l.strip()), "")
        defs.append({"type": "function", "function": {"name": safe, "description": desc, "parameters": params}})
    return defs, name_map


def to_anthropic_tools(openai_defs):
    """OpenAI function defs -> Anthropic tool defs (same safe names)."""
    return [
        {"name": d["function"]["name"], "description": d["function"]["description"],
         "input_schema": d["function"]["parameters"]}
        for d in openai_defs
    ]


async def call_protected(wrapped, name, args):
    """Call a tool through AgntID; returns (text_for_model, denied)."""
    try:
        result = await wrapped.call_tool_async(name, args)
    except Exception as exc:
        from agntid.denial import is_denial_error, parse_denial_from_exception
        if is_denial_error(exc):
            msg, _ = parse_denial_from_exception(exc)
            return msg, True
        return f"Error: {exc}", False
    if getattr(result, "denied", False):
        return result.message, True
    if hasattr(result, "value"):
        result = result.value
    return json.dumps(result, default=str) if isinstance(result, (dict, list)) else str(result), False


def direct_servers() -> dict[str, dict]:
    """The tool servers the AgntID runtime fronts, reached directly (the BEFORE run)."""
    servers = {"crowd": {"url": "http://localhost:9101/mcp"}}
    if os.environ.get("SLACK_BOT_TOKEN", "").startswith("xoxb-"):
        servers["slack"] = {"url": "http://localhost:9100/mcp"}
    url, key = os.environ.get("ELARIS_MCP_URL", ""), os.environ.get("ELARIS_API_KEY", "")
    if url.startswith("http") and key and not key.startswith("paste-"):
        header = os.environ.get("ELARIS_AUTH_HEADER", "Authorization")
        servers["elaris"] = {"url": url, "headers": {header: f"Bearer {key}" if header == "Authorization" else key}}
    return servers


def result_text(result) -> str:
    """Every text block of an MCP result, minus AgntID's internal fields.

    The SDK's cleaner keeps only the first block, which drops e.g. ElarisLabs' image URLs.
    """
    parts = []
    for block in result.content:
        text = getattr(block, "text", None)
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except ValueError:
            parts.append(text)
            continue
        clean = strip_platform_fields(parsed) if isinstance(parsed, dict) else parsed
        parts.append(json.dumps(clean, default=str, ensure_ascii=False))
    structured = getattr(result, "structured_content", None)
    if isinstance(structured, dict):  # e.g. ElarisLabs puts image.url here
        parts.append(json.dumps(strip_platform_fields(structured), default=str, ensure_ascii=False))
    return "\n".join(parts)


class AgntidTools:
    """AFTER mode: calls go through the AgntID runtime, correlated to the open task."""

    def __init__(self, client, task_id: str):
        self.client, self.task_id = client, task_id

    async def call_tool_async(self, name, args):
        return result_text(await self.client.call_tool(name, {**args, "_task_id": self.task_id}))


class DirectTools:
    """BEFORE mode: call each tool server directly, named <prefix>_<tool> like AgntID does."""

    def __init__(self, clients: dict[str, Client]):
        self.clients = clients

    async def list_tools(self):
        tools = []
        for prefix, client in self.clients.items():
            tools += [t.model_copy(update={"name": f"{prefix}_{t.name}"}) for t in await client.list_tools()]
        return tools

    async def call_tool_async(self, name, args):
        prefix, _, tool = name.partition("_")
        return result_text(await self.clients[prefix].call_tool(tool, args))


async def run_direct(prompt: str) -> None:
    async with contextlib.AsyncExitStack() as stack:
        clients = {}
        for prefix, cfg in direct_servers().items():
            clients[prefix] = await stack.enter_async_context(Client({"mcpServers": {prefix: cfg}}))
        direct = DirectTools(clients)
        tools, name_map = to_openai_tools(await direct.list_tools())
        MODE["name"] = "before"
        banner("BEFORE: PreFlight with direct tool access (no AgntID)", list(name_map.values()))
        loop = claude_loop if PROVIDER == "anthropic" else openrouter_loop
        await loop(prompt, tools, name_map, direct)


async def run(prompt: str) -> None:
    async with agntid.AgntidMCPClient(
        os.environ["AGNTID_MCP_URL"],
        access_token=os.getenv("AGNTID_ACCESS_TOKEN") or None,
    ) as client:
        visible = await agntid.get_tools_list(client)
        task_id = agntid.create_task(AGENT_ID, USER_ID, prompt)
        wrapped = AgntidTools(client, task_id)

        async with agntid.task_context(client, task_id):
            task_open = await agntid.send_task_open_details(client, task_id, AGENT_ID, USER_ID, prompt)
            if not task_open.ok:
                sys.exit(f"[agntid] task rejected: {task_open.reason}")

            # Only give the model the tools AgntID shortlisted for this prompt (fails closed).
            task_tools = agntid.filter_tools_for_task(visible, task_open)
            tools, name_map = to_openai_tools(task_tools)
            banner("AFTER: the same PreFlight agent through the AgntID runtime", list(name_map.values()))

            loop = claude_loop if PROVIDER == "anthropic" else openrouter_loop
            await loop(prompt, tools, name_map, wrapped)


async def run_tool(wrapped, name_map, safe_name, args):
    name = name_map.get(safe_name, safe_name)
    text, denied = await call_protected(wrapped, name, args)
    call = describe_call(name, args)
    if denied:
        reason = text.replace("Tool call denied: ", "").replace("Policy denied: ", "").split(" Policy: ")[0]
        print(f"  {paint('red', '⛔ DENIED ')} {call}\n             {paint('red', 'AgntID: ' + reason)}")
    elif text.startswith(("Error", "Input validation error")):
        print(f"  {paint('yellow', '⚠️  ERROR  ')} {call}\n             {paint('dim', text[:160])}")
    elif MODE["name"] == "before":
        print(f"  {paint('yellow', '▶ RAN     ')} {call}\n             {paint('dim', describe_result(name, text))}")
    else:
        print(f"  {paint('green', '✅ ALLOWED')} {call}\n             {paint('dim', describe_result(name, text))}")
    return text, denied


async def openrouter_loop(prompt, tools, name_map, wrapped):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    for _ in range(MAX_STEPS):
        resp = chat(messages, tools=tools or None)
        msg = resp.choices[0].message
        if not msg.tool_calls:
            print(paint("bold", f"\n━━━ Director report ({resp.model}) ━━━\n") + (msg.content or ""))
            return
        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            text, _ = await run_tool(wrapped, name_map, tc.function.name, json.loads(tc.function.arguments or "{}"))
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": text})
    print("[agent] stopped: step limit reached")


async def claude_loop(prompt, tools, name_map, wrapped):
    claude_tools = to_anthropic_tools(tools)
    messages = [{"role": "user", "content": prompt}]
    for _ in range(MAX_STEPS):
        resp = await claude_create(system=SYSTEM_PROMPT, messages=messages, tools=claude_tools)
        if resp.stop_reason == "refusal":
            print(f"[{resp.model}] refused: {resp.stop_details}")
            return
        tool_uses = [b for b in resp.content if b.type == "tool_use"]
        if not tool_uses:
            text = "".join(b.text for b in resp.content if b.type == "text")
            print(paint("bold", f"\n━━━ Director report ({resp.model}) ━━━\n") + text)
            return
        # Append full content (thinking/fallback blocks included) to keep the history intact.
        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for block in tool_uses:
            text, denied = await run_tool(wrapped, name_map, block.name, block.input)
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": text, "is_error": denied})
        messages.append({"role": "user", "content": results})  # all results in one message
    print("[agent] stopped: step limit reached")


if __name__ == "__main__":
    args = sys.argv[1:]
    direct = "--direct" in args
    prompt = " ".join(a for a in args if a != "--direct") or "Use demo.add_numbers to add 11 and 21"
    asyncio.run(run_direct(prompt) if direct else run(prompt))
