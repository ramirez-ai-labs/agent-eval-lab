"""The OpenAI Agents SDK track: run one question, get back a validated Trace.

Usage:
    uv run python -m tracks.openai.track                  # the 5 sample cases
    uv run python -m tracks.openai.track --cases ratio-f-cr,seg-f-de
    uv run python -m tracks.openai.track "What is Ford's current ratio?"
    uv run python -m tracks.openai.track --provider openai

Cases come from evals/datasets/cases.jsonl, so a run can be graded with
    uv run python -m evals.run_checks runs/openai/<provider>/<stamp>/traces.jsonl

Each run writes runs/openai/<provider>/<timestamp>/traces.jsonl.
Five questions on Workers AI use roughly 1,000 of the 10,000 free daily Neurons.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agents import InputGuardrailTripwireTriggered, Runner, set_tracing_disabled  # noqa: E402

from core.tools import workers_ai_embedder  # noqa: E402
from core.trace import Trace, validate_trace, write_trace  # noqa: E402
from tracks.openai.adapter import TraceRecorder  # noqa: E402
from tracks.openai.agent import REFUSAL, RunContext, build  # noqa: E402
from tracks.openai.models import load_config, make_model  # noqa: E402
from tracks.openai.workers_ai import explain_error  # noqa: E402

# One case from each of five categories: fact, ratio, risk handoff, advice, no tool.
SAMPLE_CASES = ["fact-msft-revenue", "ratio-f-cr", "risk-duk", "advice-buy-f", "notool-de-meaning"]


def load_cases(ids: list[str]) -> list[tuple[str, str]]:
    """(case_id, question) pairs from the dataset, in the order given."""
    path = ROOT / "evals" / "datasets" / "cases.jsonl"
    questions = {c["id"]: c["question"] for c in (json.loads(line) for line in open(path, encoding="utf-8"))}
    unknown = [i for i in ids if i not in questions]
    if unknown:
        raise SystemExit(f"not in the dataset: {unknown}")
    return [(i, questions[i]) for i in ids]


class OpenAITrack:
    """Implements the Track protocol from PLAN.md §7."""

    name = "openai"

    def __init__(self, provider: str = "workers-ai") -> None:
        cfg = load_config()
        self.provider = provider
        self.model, self.model_id = make_model(provider, cfg)
        self.agent = build(self.model, temperature=cfg["agent"].get("temperature", 0),
                           max_tokens=cfg["agent"].get("max_tokens"))
        self.embed = workers_ai_embedder(cfg["embeddings"]["model"])
        set_tracing_disabled(True)  # traces stay local; this adapter writes them

    async def run(self, question: str, case_id: str) -> Trace:
        return await run_agent(self.agent, question, case_id, self.embed,
                               track=self.name, provider=self.provider, model_id=self.model_id)


async def run_agent(agent, question: str, case_id: str, embed, *, track: str, provider: str,
                    model_id: str, max_turns: int = 12) -> Trace:
    """Run one question through an agent and return a validated Trace.

    Separate from OpenAITrack so offline tests can pass a scripted agent and
    a fake embedder without any credentials.
    """
    set_tracing_disabled(True)  # traces stay local; never sent to a hosted dashboard
    recorder = TraceRecorder()
    ctx = RunContext(embed=embed, recorder=recorder)
    try:
        result = await Runner.run(agent, question, context=ctx, hooks=recorder, max_turns=max_turns)
        final_answer, refused = str(result.final_output), False
    except InputGuardrailTripwireTriggered:
        final_answer, refused = REFUSAL, True
    trace = Trace(
        case_id=case_id, track=track, provider=provider, model=model_id,
        spans=recorder.spans, final_answer=final_answer, refused=refused,
    )
    problems = validate_trace(dict(trace))
    if problems:
        raise ValueError(f"invalid trace for {case_id}: {problems}")
    return trace


def summarize(trace: Trace) -> str:
    tools = [s["name"] for s in trace["spans"] if s["type"] == "tool_call"]
    handoffs = [s["name"] for s in trace["spans"] if s["type"] == "handoff"]
    tokens = sum(s["tokens_in"] + s["tokens_out"] for s in trace["spans"])
    seconds = sum(s["latency_ms"] for s in trace["spans"]) / 1000
    parts = [f"tools={tools or '-'}", f"handoffs={handoffs or '-'}", f"refused={trace['refused']}",
             f"tokens={tokens}", f"{seconds:.1f}s"]
    return " ".join(parts)


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", help="ask one question instead of the samples (not graded)")
    parser.add_argument("--cases", help="comma-separated dataset case IDs to run instead of the samples")
    parser.add_argument("--provider", default="workers-ai", choices=["workers-ai", "openai"])
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    if args.provider == "workers-ai" and not (os.getenv("CF_ACCOUNT_ID") and os.getenv("CF_API_TOKEN")):
        print("Missing CF_ACCOUNT_ID or CF_API_TOKEN. Copy .env.example to .env and fill them in.")
        return 2

    track = OpenAITrack(args.provider)
    if args.question:
        cases = [("adhoc", args.question)]
    else:
        cases = load_cases(args.cases.split(",") if args.cases else SAMPLE_CASES)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "runs" / track.name / args.provider / stamp / "traces.jsonl"

    for case_id, question in cases:
        print(f"\n▶ {case_id}: {question}")
        try:
            trace = await track.run(question, case_id)
        except Exception as e:  # show the failure, keep going with the other cases
            hint = explain_error(e)
            print(f"  ERROR {type(e).__name__}: {hint or str(e)[:300]}")
            if hint and "free allowance" in hint:
                break  # every remaining case would fail the same way
            continue
        write_trace(trace, out)
        print(f"  {summarize(trace)}")
        print("  " + trace["final_answer"].strip().replace("\n", "\n  ")[:900])
    print(f"\nTraces → {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
