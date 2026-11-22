"""The real agent, tools, guardrail, handoff and trace adapter, with a scripted
model instead of an API. No credentials or network needed.
"""

import asyncio

import pytest

from core.tools import _load_index
from tracks.openai.agent import REFUSAL, build
from tracks.openai.scripted_model import ScriptedModel
from tracks.openai.track import run_agent


def passage_embedder(passage_id: str):
    """A fake query embedder that returns one passage's own vector."""
    vectors, passages = _load_index()
    vec = vectors[[p["id"] for p in passages].index(passage_id)]
    return lambda _query: vec


def run(agent, question, embed=lambda _q: _load_index()[0][0]):
    return asyncio.run(run_agent(agent, question, "test", embed,
                                 track="openai", provider="mock", model_id="scripted"))


def spans(trace, kind):
    return [s for s in trace["spans"] if s["type"] == kind]


def test_ratio_question_uses_retrieve_then_ratio_calc():
    main = ScriptedModel([
        {"tool": "retrieve", "args": {"query": "Total current assets Total current liabilities", "ticker": "F"}},
        {"tool": "ratio_calc", "args": {"ratio": "current_ratio", "numerator": 124474, "denominator": 106859}},
        {"text": "Ford's FY2025 current ratio is 1.16 [F-FY2025-item8-017]."},
    ])
    agent = build(main, guardrail_model=ScriptedModel([{"text": "OK"}]))
    trace = run(agent, "What is Ford's current ratio?", passage_embedder("F-FY2025-item8-017"))

    tools = spans(trace, "tool_call")
    assert [t["name"] for t in tools] == ["retrieve", "ratio_calc"]
    assert tools[0]["input"]["ticker"] == "F"
    assert all(pid.startswith("F-") for pid in tools[0]["output"]["passage_ids"])
    assert tools[1]["output"] == {"ratio": "current_ratio", "value": 1.16}
    assert spans(trace, "guardrail")[0]["output"]["tripped"] is False
    assert not trace["refused"] and "1.16" in trace["final_answer"]
    assert len(spans(trace, "llm")) == 3


def test_guardrail_refuses_before_any_tool_runs():
    main = ScriptedModel([])  # would fail if the agent got a turn
    agent = build(main, guardrail_model=ScriptedModel([{"text": "ADVICE"}]))
    trace = run(agent, "Should I buy Ford stock?")

    assert trace["refused"] is True
    assert trace["final_answer"] == REFUSAL
    assert spans(trace, "guardrail")[0]["output"]["tripped"] is True
    assert spans(trace, "tool_call") == [] and spans(trace, "llm") == []


def test_risk_question_hands_off_to_risk_agent():
    main = ScriptedModel([{"tool": "transfer_to_risk_agent", "args": {}}])
    risk = ScriptedModel([
        {"tool": "retrieve", "args": {"query": "risk factors", "ticker": "DUK"}},
        {"text": "Top risks: 1) ... [DUK-FY2025-item1a-022]"},
    ])
    agent = build(main, guardrail_model=ScriptedModel([{"text": "OK"}]), risk_model=risk)
    trace = run(agent, "Give me a risk summary for Duke Energy.", passage_embedder("DUK-FY2025-item1a-022"))

    handoffs = spans(trace, "handoff")
    assert [h["name"] for h in handoffs] == ["risk_agent"]
    assert handoffs[0]["input"]["from"] == "credit_research_agent"
    assert [s["name"] for s in spans(trace, "llm")] == ["credit_research_agent", "risk_agent", "risk_agent"]
    assert "DUK-FY2025-item1a-022" in spans(trace, "tool_call")[0]["output"]["passage_ids"]


def test_no_tool_question_calls_no_tools():
    main = ScriptedModel([{"text": "Debt-to-equity compares a company's debt with its equity."}])
    agent = build(main, guardrail_model=ScriptedModel([{"text": "OK"}]))
    trace = run(agent, "What does the debt-to-equity ratio measure?")
    assert spans(trace, "tool_call") == []
    assert not trace["refused"]


def test_script_running_out_is_reported():
    agent = build(ScriptedModel([{"tool": "ratio_calc", "args": {"ratio": "net_margin", "numerator": 1, "denominator": 10}}]),
                  guardrail_model=ScriptedModel([{"text": "OK"}]))
    with pytest.raises(RuntimeError, match="script ran out"):
        run(agent, "What is Microsoft's net margin?")


# ------------------------------------------------------- setup error messages

from tracks.openai.workers_ai import explain_error  # noqa: E402


class _FakeError(Exception):
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


def test_explains_used_up_allowance():
    hint = explain_error(_FakeError("Error code: 429 - AiError: you have used up your daily free allocation of 10,000 neurons (code 4006)", 429))
    assert "resets at 00:00 UTC" in hint and "mock track" in hint


def test_explains_token_without_workers_ai_permission():
    assert "Workers AI permission" in explain_error(_FakeError("Error code: 401", 401))


def test_unknown_errors_pass_through():
    assert explain_error(_FakeError("something else", 500)) is None
