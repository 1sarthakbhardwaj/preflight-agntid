"""Slack MCP server for the creative agent.

Runs on the host; the AgntID runtime reaches it at
http://host.docker.internal:9100/mcp (the "before" demo connects directly at
http://localhost:9100/mcp). Uses the Slack app's bot token (SLACK_BOT_TOKEN in .env).

Channel arguments are plain names (e.g. "agntid-demo-review") so AgntID policies
can allowlist them readably.

Run:
    agntid-python/.venv/bin/python slack_mcp.py
"""

import os

import httpx
from fastmcp import FastMCP

from env import load_env

load_env()

SLACK_API = "https://slack.com/api/"
mcp = FastMCP("slack")


def _slack(method: str, *, get: bool = False, **params) -> dict:
    headers = {"Authorization": f"Bearer {os.environ['SLACK_BOT_TOKEN']}"}
    if get:
        resp = httpx.get(SLACK_API + method, headers=headers, params=params, timeout=20)
    else:
        resp = httpx.post(SLACK_API + method, headers=headers, json=params, timeout=20)
    data = resp.json()
    if not data.get("ok"):
        raise RuntimeError(f"Slack {method} failed: {data.get('error')}")
    return data


def _channel_id(name: str) -> str:
    name = name.lstrip("#")
    cursor = ""
    while True:
        try:
            data = _slack("conversations.list", get=True, types="public_channel",
                          exclude_archived="true", limit=1000, cursor=cursor)
        except RuntimeError as exc:
            if "missing_scope" in str(exc):  # no channels:read; chat.postMessage accepts "#name"
                return f"#{name}"
            raise
        for ch in data["channels"]:
            if ch["name"] == name:
                return ch["id"]
        cursor = data.get("response_metadata", {}).get("next_cursor", "")
        if not cursor:
            raise ValueError(f"Channel #{name} not found")


@mcp.tool
def list_channels() -> list[str]:
    """List public Slack channel names the bot can see."""
    data = _slack("conversations.list", get=True, types="public_channel",
                  exclude_archived="true", limit=1000)
    return [ch["name"] for ch in data["channels"]]


@mcp.tool
def read_channel(channel: str, limit: int = 10) -> list[dict]:
    """Read the most recent messages in a Slack channel (e.g. a creative brief)."""
    data = _slack("conversations.history", get=True, channel=_channel_id(channel), limit=limit)
    return [{"user": m.get("user", ""), "text": m.get("text", "")} for m in data["messages"]]


@mcp.tool
def post_message(channel: str, text: str, image_urls: list[str] | None = None) -> dict:
    """Post a message (optionally with images) to a Slack channel by name."""
    # Slack caps a section block at 3000 chars; long reports go as plain message text.
    blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": text}}] if len(text) <= 2900 else []
    blocks += [{"type": "image", "image_url": url, "alt_text": "creative"} for url in image_urls or []]
    data = _slack("chat.postMessage", channel=_channel_id(channel), text=text, **({"blocks": blocks} if blocks else {}))
    return {"ok": True, "channel": channel, "ts": data["ts"]}


@mcp.tool
def send_dm(user_id: str, text: str) -> dict:
    """Send a direct message to a Slack user by user ID."""
    dm = _slack("conversations.open", users=user_id)
    data = _slack("chat.postMessage", channel=dm["channel"]["id"], text=text)
    return {"ok": True, "user_id": user_id, "ts": data["ts"]}


if __name__ == "__main__":
    mcp.run(transport="http", host="127.0.0.1", port=int(os.environ.get("SLACK_MCP_PORT", "9100")))
