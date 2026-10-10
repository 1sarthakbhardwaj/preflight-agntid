"""LLM clients for the agent.

- OpenRouter (OpenAI format): DeepSeek V4 Pro primary, Claude Haiku 5.5 fallback.
- Anthropic (native SDK): Claude Opus 5.5 on your own API key.

LLM_PROVIDER in .env picks which one agent.py uses (anthropic | openrouter).

Usage:
    from llm import chat, claude_create
    resp = chat([{"role": "user", "content": "hi"}], tools=openai_tools)
    msg = await claude_create(system=..., messages=..., tools=anthropic_tools)
"""

import os

import anthropic
from openai import OpenAI

from env import load_env

load_env()

PROVIDER = os.environ.get("LLM_PROVIDER", "openrouter")
PRIMARY_MODEL = os.environ.get("LLM_PRIMARY_MODEL", "deepseek/deepseek-v4-pro")
FALLBACK_MODEL = os.environ.get("LLM_FALLBACK_MODEL", "anthropic/claude-haiku-5.5")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-5-5")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY", ""),
)
# Reads ANTHROPIC_API_KEY. Org-level keys (not scoped to a workspace) must name
# the workspace to bill via the anthropic-workspace-id header.
_workspace = os.environ.get("ANTHROPIC_WORKSPACE_ID")
claude = anthropic.AsyncAnthropic(
    default_headers={"anthropic-workspace-id": _workspace} if _workspace else None,
)


async def claude_create(*, system, messages, tools=None):
    """One Claude turn. Adaptive thinking is always on for Opus 5.5; effort is set
    explicitly (its default is medium). Server-side fallback re-runs a refused
    request on a suitable model automatically."""
    kwargs = {"tools": tools} if tools else {}
    return await claude.beta.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=16000,
        system=system,
        messages=messages,
        output_config={"effort": "high"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        **kwargs,
    )


def chat(messages, tools=None, **kwargs):
    """Chat completion with OpenRouter-native fallback (primary, then fallback)."""
    if tools:
        kwargs["tools"] = tools
    return client.chat.completions.create(
        model=PRIMARY_MODEL,
        messages=messages,
        extra_body={"models": [PRIMARY_MODEL, FALLBACK_MODEL]},
        **kwargs,
    )


if __name__ == "__main__":
    import asyncio

    if PROVIDER == "anthropic":
        m = asyncio.run(claude_create(system="Be brief.", messages=[{"role": "user", "content": "Reply with just: ok"}]))
        text = next((b.text for b in m.content if b.type == "text"), None)
        print(f"model used: {m.model} -> {text!r} (stop_reason={m.stop_reason})")
    else:
        r = chat([{"role": "user", "content": "Reply with just: ok"}], max_tokens=300)
        print(f"model used: {r.model} -> {r.choices[0].message.content!r}")
