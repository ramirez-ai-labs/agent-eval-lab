"""OpenAI client for Workers AI, with one compatibility fix.

Workers AI's Chat Completions endpoint rejects assistant messages whose
"content" is null. That's the standard OpenAI format for a message that
only contains tool calls, and the Agents SDK sends it after every tool call.
This client rewrites null content to "" before the request goes out.

It also meters cost: after each Chat Completions response it reads
`usage.neurons` and adds it to the current run's meter (core/cost.py).

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

from core import cost

BASE_URL = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"


def _fix_null_content(body: dict) -> bool:
    changed = False
    for msg in body.get("messages", []):
        if msg.get("role") == "assistant" and msg.get("content") is None:
            msg["content"] = ""
            changed = True
    return changed


class _CompatTransport(httpx.AsyncBaseTransport):
    def __init__(self, inner: httpx.AsyncBaseTransport | None = None) -> None:
        self._inner = inner or httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/chat/completions") and request.content:
            body = json.loads(request.content)
            if _fix_null_content(body):
                content = json.dumps(body).encode()
                headers = {k: v for k, v in request.headers.items() if k.lower() != "content-length"}
                request = httpx.Request(request.method, request.url, headers=headers, content=content)
        response = await self._inner.handle_async_request(request)
        if request.url.path.endswith("/chat/completions") and response.status_code == 200:
            response = await _metered(response, request)
        return response

    async def aclose(self) -> None:
        await self._inner.aclose()


async def _metered(response: httpx.Response, request: httpx.Request) -> httpx.Response:
    """Read usage.neurons from the body, then hand back an equivalent response."""
    body = await response.aread()  # decoded, so drop the encoding headers below
    try:
        cost.record(cost.neurons_in(json.loads(body)))
    except ValueError:
        pass  # not JSON; the client will report it
    headers = [(k, v) for k, v in response.headers.items()
               if k.lower() not in ("content-encoding", "content-length", "transfer-encoding")]
    return httpx.Response(response.status_code, headers=headers, content=body,
                          request=request, extensions=response.extensions)


def explain_error(error: Exception) -> str | None:
    """A plain-language explanation for common Workers AI setup errors, or None.

    These are the errors learners actually hit, so the lab explains them
    instead of showing a bare status code.
    """
    # The OpenAI client's error text is often just "Error code: 429"; Cloudflare's
    # actual message is in the response body, so read that too.
    response = getattr(error, "response", None)
    parts = [str(error), str(getattr(error, "body", "") or "")]
    try:
        parts.append(response.text if response is not None else "")
    except Exception:
        pass
    text = " ".join(parts)
    status = getattr(error, "status_code", None) or getattr(response, "status_code", None)
    if "4006" in text or "daily free allocation" in text:
        return ("The Workers AI free allowance (10,000 Neurons per day) is used up. It resets at "
                "00:00 UTC. Until then, use the mock track: it runs offline and costs nothing.")
    if status == 401 or "Authentication error" in text:
        return ("Cloudflare rejected the API token. Check that CF_API_TOKEN in .env is a token with the "
                "Workers AI permission (Read and Edit). A valid token without that permission also fails this way.")
    if status == 403:
        return "The token is valid but can't access this account. Check that CF_ACCOUNT_ID matches the token's account."
    return None


def make_client(account_id: str, api_token: str) -> AsyncOpenAI:
    """An AsyncOpenAI client pointed at Workers AI, with the compatibility fix."""
    return AsyncOpenAI(
        base_url=BASE_URL.format(account_id=account_id),
        api_key=api_token,
        http_client=httpx.AsyncClient(transport=_CompatTransport(), timeout=120),
    )
