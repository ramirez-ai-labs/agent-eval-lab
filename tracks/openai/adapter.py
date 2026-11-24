"""Trace adapter: Agents SDK events → the common Trace format in core/trace.py.

Every check reads traces in that format, so this file is what lets the same
eval suite judge this track and every other one.
"""

from __future__ import annotations

import json
import time
from typing import Any

from agents import RunHooks

from core.cost import Meter
from core.trace import Span


def _parse(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError):
        return {"value": str(value)[:500]}
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list):
        return {"items": parsed}
    return {"value": parsed}


def _summarize_tool_output(name: str, raw: Any) -> dict:
    out = _parse(raw)
    if name == "retrieve":
        # Keep traces small: IDs (for Recall@k) and sources, not full passage text.
        items = out.get("items", [])
        return {"passage_ids": [p["id"] for p in items], "sources": [p.get("source", "") for p in items]}
    return out


class TraceRecorder(RunHooks):
    """Collects spans while the SDK runs."""

    def __init__(self) -> None:
        self.spans: list[Span] = []
        self._llm_start: dict[str, float] = {}
        self._tool_start: dict[str, tuple[float, dict]] = {}
        self._last_llm_id: str | None = None
        self.meter = Meter()  # Neurons, on providers that report them (core/cost.py)

    def _id(self) -> str:
        return f"s{len(self.spans) + 1}"

    # model calls
    async def on_llm_start(self, context, agent, system_prompt, input_items) -> None:
        self._llm_start[agent.name] = time.perf_counter()

    async def on_llm_end(self, context, agent, response) -> None:
        start = self._llm_start.pop(agent.name, time.perf_counter())
        calls = [getattr(o, "name", None) for o in response.output if getattr(o, "type", "") == "function_call"]
        output: dict[str, Any] = {"tool_calls": [c for c in calls if c]}
        neurons = self.meter.take()
        if neurons is not None:
            output["neurons"] = neurons
        span_id = self._id()
        self.spans.append(Span(
            id=span_id, parent_id=None, type="llm", name=agent.name,
            input={},
            output=output,
            tokens_in=int(response.usage.input_tokens or 0),
            tokens_out=int(response.usage.output_tokens or 0),
            latency_ms=int((time.perf_counter() - start) * 1000),
        ))
        self._last_llm_id = span_id

    # tool calls
    async def on_tool_start(self, context, agent, tool) -> None:
        call_id = getattr(context, "tool_call_id", None) or tool.name
        args = _parse(getattr(context, "tool_arguments", "{}"))
        self._tool_start[call_id] = (time.perf_counter(), args)

    async def on_tool_end(self, context, agent, tool, result) -> None:
        call_id = getattr(context, "tool_call_id", None) or tool.name
        start, args = self._tool_start.pop(call_id, (time.perf_counter(), {}))
        self.spans.append(Span(
            id=self._id(), parent_id=self._last_llm_id, type="tool_call", name=tool.name,
            input=args, output=_summarize_tool_output(tool.name, result),
            tokens_in=0, tokens_out=0,
            latency_ms=int((time.perf_counter() - start) * 1000),
        ))

    # handoffs
    async def on_handoff(self, context, from_agent, to_agent) -> None:
        self.spans.append(Span(
            id=self._id(), parent_id=self._last_llm_id, type="handoff", name=to_agent.name,
            input={"from": from_agent.name}, output={},
            tokens_in=0, tokens_out=0, latency_ms=0,
        ))

    # guardrail (recorded by the runner, from the guardrail's decision)
    def add_guardrail(self, name: str, decision: dict, latency_ms: int) -> None:
        self.spans.append(Span(
            id=self._id(), parent_id=None, type="guardrail", name=name,
            input={}, output=self._with_neurons({"tripped": bool(decision.get("tripped")),
                                                 "verdict": decision.get("verdict", "")}),
            tokens_in=int(decision.get("tokens_in", 0)), tokens_out=int(decision.get("tokens_out", 0)),
            latency_ms=latency_ms,
        ))

    def _with_neurons(self, output: dict) -> dict:
        neurons = self.meter.take()  # the classifier's model call, made just before this span
        return output if neurons is None else {**output, "neurons": neurons}

    def total_neurons(self) -> float | None:
        """Neurons for the whole run, or None if the provider reported none."""
        return round(self.meter.total, 2) if self.meter.calls else None
