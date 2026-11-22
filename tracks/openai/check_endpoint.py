"""Check that the OpenAI Agents SDK works against Workers AI.

This is the first M1 task and the plan's biggest open risk. Workers AI
supports the Responses API only for gpt-oss models, and only without
streaming. If the SDK's requests work there, the free tier and the OpenAI
tier share one code path. If they don't, Workers AI falls back to the Chat
Completions model class.

The script runs two small tests on each API:
  1. tool call: does the agent call ratio_calc instead of doing the math?
  2. handoff:   does the main agent hand off to a second agent?

Usage:
    uv run python tracks/openai/check_endpoint.py            # both APIs
    uv run python tracks/openai/check_endpoint.py --api responses
    uv run python tracks/openai/check_endpoint.py --model @cf/openai/gpt-oss-20b

Needs CF_ACCOUNT_ID and CF_API_TOKEN in .env. Each full run uses a few
hundred Neurons of the 10,000 free daily allowance.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import AsyncOpenAI

from agents import (
    Agent,
    HandoffOutputItem,
    OpenAIChatCompletionsModel,
    OpenAIResponsesModel,
    Runner,
    ToolCallItem,
    function_tool,
    set_tracing_disabled,
)

ROOT = Path(__file__).resolve().parents[2]


@function_tool
def ratio_calc(ratio: str, numerator: float, denominator: float) -> dict:
    """Compute a financial ratio. Always use this tool for ratio math.

    Args:
        ratio: The ratio name, e.g. "debt_to_equity".
        numerator: The numerator, e.g. total debt.
        denominator: The denominator, e.g. total shareholders' equity.
    """
    return {"ratio": ratio, "value": round(numerator / denominator, 2)}


def load_config() -> dict:
    with open(ROOT / "config" / "models.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_model(api: str, model_id: str, client: AsyncOpenAI):
    if api == "responses":
        return OpenAIResponsesModel(model=model_id, openai_client=client)
    return OpenAIChatCompletionsModel(model=model_id, openai_client=client)


async def check_tool_call(model) -> tuple[bool, str]:
    agent = Agent(
        name="credit_research_agent",
        instructions=(
            "You answer questions about company financials. "
            "Always use the ratio_calc tool for any ratio. Never do the math yourself."
        ),
        tools=[ratio_calc],
        model=model,
    )
    result = await Runner.run(
        agent,
        "Total debt is 80000 and total shareholders' equity is 50000. "
        "What is the debt-to-equity ratio?",
        max_turns=4,
    )
    called = [
        i for i in result.new_items
        if isinstance(i, ToolCallItem) and getattr(i.raw_item, "name", "") == "ratio_calc"
    ]
    answer = str(result.final_output).strip().replace("\n", " ")
    ok = bool(called) and "1.6" in answer
    detail = f"tool called: {bool(called)}; answer: {answer[:120]}"
    return ok, detail


async def check_handoff(model) -> tuple[bool, str]:
    risk_agent = Agent(
        name="risk_agent",
        instructions="You write a two-sentence risk summary for the company named.",
        model=model,
    )
    main_agent = Agent(
        name="credit_research_agent",
        instructions=(
            "You answer factual questions about companies. "
            "If the user asks for a risk summary, hand off to risk_agent."
        ),
        handoffs=[risk_agent],
        model=model,
    )
    result = await Runner.run(
        main_agent,
        "Give me a risk summary for a highly leveraged utility company.",
        max_turns=4,
    )
    handed_off = any(isinstance(i, HandoffOutputItem) for i in result.new_items)
    ok = handed_off and result.last_agent.name == "risk_agent"
    detail = f"handoff: {handed_off}; last agent: {result.last_agent.name}"
    return ok, detail


async def run_api(api: str, model_id: str, client: AsyncOpenAI) -> dict[str, bool]:
    print(f"\n== {api} API · {model_id}")
    model = make_model(api, model_id, client)
    results: dict[str, bool] = {}
    for name, check in (("tool call", check_tool_call), ("handoff", check_handoff)):
        start = time.perf_counter()
        try:
            ok, detail = await check(model)
        except Exception as e:  # report the endpoint's error, don't hide it
            ok, detail = False, f"{type(e).__name__}: {str(e)[:300]}"
        ms = int((time.perf_counter() - start) * 1000)
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<10} {ms:>6} ms  {detail}")
        results[name] = ok
    return results


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--api", choices=["responses", "chat", "both"], default="both")
    parser.add_argument("--model", help="override the model from config/models.yaml")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    account_id = os.getenv("CF_ACCOUNT_ID")
    token = os.getenv("CF_API_TOKEN")
    if not account_id or not token:
        print("Missing CF_ACCOUNT_ID or CF_API_TOKEN. Copy .env.example to .env and fill them in.")
        return 2

    cfg = load_config()
    model_id = args.model or cfg["agent"]["workers_ai_model"]
    base_url = cfg["agent"]["base_url"].replace("${CF_ACCOUNT_ID}", account_id)

    set_tracing_disabled(True)  # traces stay local; no hosted exporter
    client = AsyncOpenAI(base_url=base_url, api_key=token)

    apis = ["responses", "chat"] if args.api == "both" else [args.api]
    summary = {api: await run_api(api, model_id, client) for api in apis}

    print("\n== Result")
    if all(summary.get("responses", {}).values()) and "responses" in summary:
        print("  Responses API works: Workers AI and OpenAI share one code path.")
        return 0
    if all(summary.get("chat", {}).values()) and "chat" in summary:
        print("  Responses API failed, Chat Completions works: use the Chat Completions model class for Workers AI.")
        return 1
    print("  Neither API passed every check. See the errors above.")
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
