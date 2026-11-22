"""OpenAI client for Workers AI, with one compatibility fix.

Workers AI's Chat Completions endpoint rejects assistant messages whose
"content" is null. That's the standard OpenAI format for a message that
only contains tool calls, and the Agents SDK sends it after every tool call.
This client rewrites null content to "" before the request goes out.

Everything Workers AI-specific lives here, so the agent code stays the same
for every provider.

Checked Nov 21, 2026 against openai-agents 0.25.0 and @cf/openai/gpt-oss-120b.
Also found: the Responses API on Workers AI rejects function tools (400),
so the Workers AI provider uses the Chat Completions model class.
"""

from __future__ import annotations

import json

import httpx2 as httpx
from openai import AsyncOpenAI

BASE_URL = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"


def _fix_null_content(body: dict) -> bool:
    changed = False
    for msg in body.get("messages", []):
        if msg.get("role") == "assistant" and msg.get("content") is None:
            msg["content"] = ""
            changed = True
    return changed


class _CompatTransport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self._inner = httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/chat/completions") and request.content:
            body = json.loads(request.content)
            if _fix_null_content(body):
                content = json.dumps(body).encode()
                headers = {k: v for k, v in request.headers.items() if k.lower() != "content-length"}
                request = httpx.Request(request.method, request.url, headers=headers, content=content)
        return await self._inner.handle_async_request(request)

    async def aclose(self) -> None:
        await self._inner.aclose()


def make_client(account_id: str, api_token: str) -> AsyncOpenAI:
    """An AsyncOpenAI client pointed at Workers AI, with the compatibility fix."""
    return AsyncOpenAI(
        base_url=BASE_URL.format(account_id=account_id),
        api_key=api_token,
        http_client=httpx.AsyncClient(transport=_CompatTransport(), timeout=120),
    )
