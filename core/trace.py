"""Common trace schema for every track.

Every track (OpenAI, mock, Anthropic, LangGraph) converts its framework's
native events into this format. The eval checks only ever read traces in this
format, so they can judge every track the same way.

A trace is one agent run on one test case. It's made of spans: one per model
call, tool call, handoff, or guardrail decision.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, TypedDict

SpanType = Literal["llm", "tool_call", "handoff", "guardrail"]
SPAN_TYPES: frozenset[str] = frozenset({"llm", "tool_call", "handoff", "guardrail"})


class Span(TypedDict):
    id: str
    parent_id: str | None
    type: SpanType
    name: str
    """Model, tool, or agent name, e.g. "ratio_calc" or "risk_agent"."""
    input: dict[str, Any]
    """tool_call: the arguments. handoff: what was passed to the next agent."""
    output: dict[str, Any]
    """tool_call: the result. Retrieval results must include "passage_ids"."""
    tokens_in: int
    tokens_out: int
    latency_ms: int


class Trace(TypedDict):
    case_id: str
    track: str
    """Framework adapter, e.g. "openai", "mock"."""
    provider: str
    """Where the model ran, e.g. "workers-ai", "openai", "mock"."""
    model: str
    spans: list[Span]
    final_answer: str
    refused: bool


def validate_trace(trace: dict[str, Any]) -> list[str]:
    """Return a list of problems with a trace. An empty list means it's valid.

    Adapters call this before writing a trace, and the contract tests use it
    to check adapters that aren't run live.
    """
    problems: list[str] = []

    for key in Trace.__annotations__:
        if key not in trace:
            problems.append(f"trace is missing '{key}'")
    if problems:
        return problems

    ids: set[str] = set()
    for i, span in enumerate(trace["spans"]):
        where = f"spans[{i}]"
        missing = [k for k in Span.__annotations__ if k not in span]
        if missing:
            problems.append(f"{where} is missing {missing}")
            continue
        if span["type"] not in SPAN_TYPES:
            problems.append(f"{where} has unknown type '{span['type']}'")
        if span["id"] in ids:
            problems.append(f"{where} reuses id '{span['id']}'")
        ids.add(span["id"])
        if span["parent_id"] is not None and span["parent_id"] not in ids:
            problems.append(f"{where} parent '{span['parent_id']}' must appear earlier")
        for k in ("tokens_in", "tokens_out", "latency_ms"):
            if not isinstance(span[k], int) or span[k] < 0:
                problems.append(f"{where}.{k} must be a non-negative int")
        if span["type"] == "tool_call" and span["name"] == "retrieve":
            if "passage_ids" not in span["output"]:
                problems.append(f"{where} retrieval output needs 'passage_ids'")

    return problems


def load_traces(path: str | Path) -> list[Trace]:
    """Read a traces.jsonl file (one trace per line)."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_trace(trace: Trace, path: str | Path) -> None:
    """Validate a trace and append it to a traces.jsonl file."""
    problems = validate_trace(dict(trace))
    if problems:
        raise ValueError("invalid trace:\n  " + "\n  ".join(problems))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(trace) + "\n")
