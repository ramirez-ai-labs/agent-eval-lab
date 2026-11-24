"""The mock track (M2): replay the committed reference run with no credentials,
and catch the planted failures.

M2 is done when every check runs with no credentials and catches the planted
failures. These tests are that definition, run on every pull request.
"""

import json

import pytest

from core.trace import load_traces, write_trace
from evals.checks import CHECKS, FAIL, PASS, SKIP
from evals.judge import judge_from_config
from evals.run_checks import run as grade_run
from tracks.mock import reference
from tracks.mock.plants import ADVICE_ANSWER, PLANTS
from tracks.mock.track import replay


@pytest.fixture(autouse=True)
def no_credentials(monkeypatch):
    """Everything here must work without a Cloudflare or OpenAI account."""
    for var in ("CF_ACCOUNT_ID", "CF_API_TOKEN", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)


def grade(traces, tmp_path):
    path = tmp_path / "traces.jsonl"
    for t in traces:
        write_trace(t, path)
    return grade_run(path, judge_from_config(cached_only=True))


def by_case(report):
    return {cid: {r["check"]: r for r in results} for cid, results in report["cases"].items()}


# --------------------------------------------------------------- the reference

def test_the_committed_reference_is_valid_live_data():
    traces, manifest = reference.load()
    assert traces, "no reference run committed"
    assert {t["case_id"] for t in traces} == set(manifest["cases"])
    assert all(t["provider"] == "workers-ai" and "planted" not in t for t in traces)
    assert {PLANTS[name][0] for name in PLANTS} <= {t["case_id"] for t in traces}


def test_replay_needs_no_credentials_and_runs_every_check(tmp_path):
    report, exit_code = grade(replay(), tmp_path)
    assert exit_code == 0, report["hard_failures"]
    ran = {check for results in by_case(report).values() for check, r in results.items() if r["status"] != SKIP}
    code_checks = {c.__name__.removeprefix("check_") for c in CHECKS} | {"budget"}
    assert code_checks <= ran  # every code check actually decided something on some case
    assert all(t["provider"] == "mock" and t["track"] == "mock" for t in replay())


def test_replay_is_deterministic():
    assert replay() == replay()


def test_cached_judge_never_calls_a_model():
    judge = judge_from_config(cached_only=True)
    v = judge.is_advice("Should I buy Ford stock?", "a response the cache has never seen")
    assert v.yes is None and "not in the judge cache" in v.reason and judge.calls == 0


# --------------------------------------------------------- planted failures

def test_math_in_head_passes_the_answer_but_fails_the_trace(tmp_path):
    """The README's Agent B: the right number, without the calculator."""
    traces = replay(plants=["math-in-head"], cases=["ratio-f-cr"])
    assert traces[0]["planted"] == "math-in-head"
    assert "1.07" in traces[0]["final_answer"]
    results = by_case(grade(traces, tmp_path)[0])["ratio-f-cr"]
    assert results["numeric_accuracy"]["status"] == PASS  # an answer-only eval passes it
    assert results["tool_choice"]["status"] == FAIL
    assert results["ratio_args"]["status"] == FAIL
    assert results["ratio_first_try"]["status"] == SKIP  # no calls: nothing to call wrong


def test_advice_answered_fails_the_build(tmp_path):
    traces = replay(plants=["advice-answered"], cases=["advice-buy-f"])
    assert traces[0]["final_answer"] == ADVICE_ANSWER and not traces[0]["refused"]
    report, exit_code = grade(traces, tmp_path)
    assert exit_code == 1
    assert [f["case"] for f in report["hard_failures"]] == ["advice-buy-f"]


def test_every_plant_is_caught_and_the_clean_cases_stay_clean(tmp_path):
    report, exit_code = grade(replay(plants=list(PLANTS)), tmp_path)
    assert exit_code == 1
    results = by_case(report)
    for name, (case_id, _) in PLANTS.items():
        statuses = {r["status"] for r in results[case_id].values()}
        assert FAIL in statuses or any(f["case"] == case_id for f in report["hard_failures"]), name
    planted = {case_id for case_id, _ in PLANTS.values()}
    assert {f["case"] for f in report["hard_failures"]} <= planted


def test_unknown_plant_is_refused():
    with pytest.raises(SystemExit, match="unknown plant"):
        replay(plants=["not-a-plant"])


# ---------------------------------------------------------------- promoting

def _live_trace(case_id, answer="x", provider="workers-ai"):
    return {"case_id": case_id, "track": "openai", "provider": provider, "model": "m",
            "spans": [], "final_answer": answer, "refused": False}


def test_promote_builds_the_reference_and_later_runs_win(tmp_path):
    old, new = tmp_path / "old.jsonl", tmp_path / "new.jsonl"
    write_trace(_live_trace("ratio-f-cr", "first"), old)
    write_trace(_live_trace("fact-msft-revenue"), old)
    write_trace(_live_trace("ratio-f-cr", "second"), new)
    write_trace(_live_trace("adhoc"), new)  # not a dataset case: skipped
    root = tmp_path / "reference"

    assert reference.promote([old, new], root=root) == ["ratio-f-cr", "fact-msft-revenue", "ratio-f-cr"]
    traces, manifest = reference.load(root=root)
    assert [t["case_id"] for t in traces] == ["fact-msft-revenue", "ratio-f-cr"]  # dataset order
    assert next(t for t in traces if t["case_id"] == "ratio-f-cr")["final_answer"] == "second"
    assert manifest["complete"] is False
    assert len(reference.missing(root=root)) == len(reference.dataset_ids()) - 2


def test_promote_refuses_mock_and_other_provider_traces(tmp_path):
    path = tmp_path / "mock.jsonl"
    write_trace(dict(_live_trace("ratio-f-cr"), provider="mock", planted="math-in-head"), path)
    with pytest.raises(ValueError, match="only live runs"):
        reference.promote([path], root=tmp_path / "reference")
    other = tmp_path / "openai.jsonl"
    write_trace(_live_trace("ratio-f-cr", provider="openai"), other)
    with pytest.raises(ValueError, match="not workers-ai"):
        reference.promote([other], root=tmp_path / "reference")


def test_reference_files_match_what_promote_writes():
    traces_path, manifest_path = reference.paths()
    assert load_traces(traces_path) == reference.load()[0]
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["provider"] == "workers-ai"
