"""Neuron metering, offline: the Workers AI transport, the trace recorder, the
report and the judge. No credentials or network needed.

The first live run cost about 6,500 Neurons against an estimate of 1,000, so
cost is measured from each API response instead of estimated (core/cost.py).
"""

import asyncio
import gzip
import json

import httpx2 as httpx

from core import cost
from core.trace import validate_trace, write_trace
from evals.judge import Judge
from evals.run_checks import run as grade_run
from tests.test_agent_offline import passage_embedder, run, spans
from tracks.openai.agent import build
from tracks.openai.scripted_model import ScriptedModel
from tracks.openai.workers_ai import _CompatTransport

BODY = {"choices": [{"message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 43, "completion_tokens": 5, "neurons": 8.57}}


def _post(body=BODY, gzipped=False):
    raw = json.dumps(body).encode()
    headers = {"content-type": "application/json"}
    if gzipped:
        raw, headers["content-encoding"] = gzip.compress(raw), "gzip"
    inner = httpx.MockTransport(lambda req: httpx.Response(200, headers=headers, content=raw))
    t = _CompatTransport(inner)
    req = httpx.Request("POST", "https://x/ai/v1/chat/completions", json={"messages": []})

    async def go():
        resp = await t.handle_async_request(req)
        return json.loads(await resp.aread())
    return asyncio.run(go())


def test_transport_records_neurons_and_keeps_the_body():
    meter = cost.Meter()
    with cost.metering(meter):
        assert _post() == BODY
        assert _post(gzipped=True) == BODY  # decoded once, not twice
    assert meter.calls == 2 and round(meter.total, 2) == 17.14


def test_transport_outside_a_run_records_nothing():
    assert _post() == BODY  # no current meter: no error, nothing recorded


class MeteredModel(ScriptedModel):
    """A scripted model that reports Neurons per call, as Workers AI does."""

    def __init__(self, steps, neurons):
        super().__init__(steps)
        self.neurons = neurons

    async def get_response(self, *args, **kwargs):
        cost.record(self.neurons)
        return await super().get_response(*args, **kwargs)


def test_neurons_go_on_the_spans_that_spent_them():
    main = MeteredModel([
        {"tool": "retrieve", "args": {"query": "Total current assets", "ticker": "F"}},
        {"tool": "ratio_calc", "args": {"ratio": "current_ratio", "numerator": 123487, "denominator": 114890}},
        {"text": "Ford's FY2025 current ratio is 1.07 [F-FY2025-item8-017]."},
    ], neurons=40.0)
    agent = build(main, guardrail_model=MeteredModel([{"text": "OK"}], neurons=3.0))
    trace = run(agent, "What is Ford's current ratio?", passage_embedder("F-FY2025-item8-017"))

    assert [s["output"]["neurons"] for s in spans(trace, "guardrail")] == [3.0]
    assert [s["output"]["neurons"] for s in spans(trace, "llm")] == [40.0, 40.0, 40.0]
    assert trace["neurons"] == 123.0
    assert validate_trace(dict(trace)) == []


def test_providers_without_neurons_leave_them_out():
    agent = build(ScriptedModel([{"text": "It compares debt with equity."}]),
                  guardrail_model=ScriptedModel([{"text": "OK"}]))
    trace = run(agent, "What does the debt-to-equity ratio measure?")
    assert "neurons" not in trace
    assert all("neurons" not in s["output"] for s in trace["spans"])
    assert validate_trace(dict(trace)) == []


def test_validator_rejects_negative_neurons():
    agent = build(ScriptedModel([{"text": "x"}]), guardrail_model=ScriptedModel([{"text": "OK"}]))
    trace = dict(run(agent, "What does the debt-to-equity ratio measure?"), neurons=-1)
    assert "trace 'neurons' must be a non-negative number" in validate_trace(trace)


def test_report_adds_up_agent_and_judge_neurons(tmp_path):
    agent = build(MeteredModel([{"text": "It compares debt with equity."}], neurons=25.0),
                  guardrail_model=MeteredModel([{"text": "OK"}], neurons=2.0))
    trace = dict(run(agent, "What does the debt-to-equity ratio measure?"), case_id="notool-de-meaning")
    path = tmp_path / "traces.jsonl"
    write_trace(trace, path)

    def complete(system, user):
        complete.meter.add(140.0)
        return "NO\nIt explains a ratio."
    complete.meter = cost.Meter()
    judge = Judge(complete, "fake-judge", cache_path=None)

    report, _ = grade_run(path, judge)
    n = report["neurons"]
    assert n["agent"] == 27.0 and n["agent_per_case"] == {"notool-de-meaning": 27.0}
    assert n["judge"] == 140.0 * judge.calls and judge.calls >= 1


def test_report_without_neurons_says_none(tmp_path):
    agent = build(ScriptedModel([{"text": "It compares debt with equity."}]),
                  guardrail_model=ScriptedModel([{"text": "OK"}]))
    path = tmp_path / "traces.jsonl"
    write_trace(dict(run(agent, "What does the debt-to-equity ratio measure?"), case_id="notool-de-meaning"), path)
    report, _ = grade_run(path)
    assert report["neurons"]["agent"] is None and report["neurons"]["judge"] is None
