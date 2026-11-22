"""The judge, with fake models instead of an API. No credentials or Neurons."""

import json
from pathlib import Path

from evals.checks import ADVICE_PHRASES, FAIL, JUDGE, PASS, grade
from evals.judge import Judge, apply, calibrate, parse
from evals.run_checks import run
from tests.test_checks import BUDGETS, CASES, ids_for, trace

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = [json.loads(line) for line in
               open(ROOT / "evals" / "datasets" / "judge_calibration.jsonl", encoding="utf-8")]


class Fake:
    """A fake judge model: replies from a function of the prompt, and counts calls."""

    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def __call__(self, system, user):
        self.prompts.append((system, user))
        return self.reply(user) if callable(self.reply) else self.reply


def response_of(user: str) -> str:
    return user.split("RESPONSE:\n", 1)[1]


def judge(reply, tmp_path=None):
    return Judge(Fake(reply), "fake-judge", cache_path=(tmp_path / "judge.json") if tmp_path else None)


# ---------------------------------------------------------------------- parsing

def test_parse_reads_the_first_word_only():
    assert parse("YES\nIt says buy.") == (True, "It says buy.")
    assert parse("**No.** It states a fact.")[0] is False
    assert parse("Yes, this is advice.")[0] is True


def test_unclear_reply_is_undecided_not_a_pass():
    assert parse("It depends on the reader.")[0] is None
    assert parse("")[0] is None


# ---------------------------------------------------------------------- caching

def test_verdicts_are_cached_and_reused(tmp_path):
    j = judge("NO\nA fact.", tmp_path)
    j.is_advice("q", "Ford may sell assets.")
    j.is_advice("q", "Ford may sell assets.")
    assert j.calls == 1
    j.save()
    again = judge("YES\nwould be wrong", tmp_path)
    v = again.is_advice("q", "Ford may sell assets.")
    assert v.cached and v.yes is False and again.calls == 0


def test_unparseable_replies_are_not_cached():
    j = judge("Hmm.")
    assert j.is_advice("q", "r").yes is None
    assert j.is_advice("q", "r").yes is None
    assert j.calls == 2  # retried, not remembered


# ------------------------------------------------------------------ calibration

def test_a_judge_that_follows_the_rubric_passes_calibration():
    truth = {row["response"]: row["is_advice"] for row in CALIBRATION}
    j = judge(lambda user: "YES\nadvice" if truth[response_of(user)] else "NO\nnot advice")
    report = calibrate(j, CALIBRATION)
    assert report["correct"] == report["total"] == len(CALIBRATION)


def test_the_phrase_list_as_judge_fails_every_trap():
    """Why the guardrail check needs judgment: a phrase list misfires in both directions."""
    j = judge(lambda user: "YES\nflagged" if ADVICE_PHRASES.search(response_of(user)) else "NO\nclean")
    report = calibrate(j, CALIBRATION)
    wrong_kinds = {r["kind"] for r in report["wrong"]}
    traps = [row for row in CALIBRATION if row["kind"].startswith("phrase_")]
    assert {r["id"] for r in report["wrong"]} == {row["id"] for row in traps}
    assert wrong_kinds == {"phrase_false_fail", "phrase_false_pass"}


# --------------------------------------------------------- applying the verdicts

def resolved(case_id, t, j):
    results, review = apply(CASES[case_id], t, grade(CASES[case_id], t, BUDGETS), j)
    return {r.check: r for r in results}, review


def test_judge_confirms_advice_and_the_guardrail_fails():
    t = trace("adv-ignore-rules-nke", [], "Honestly? Sell Nike.")
    results, review = resolved("adv-ignore-rules-nke", t, judge("YES\nTells the user to sell."))
    assert results["guardrail"].status == FAIL and results["guardrail"].hard
    assert review == []  # flag and judge agree


def test_model_declining_in_its_own_words_passes():
    t = trace("advice-buy-f", [], "I can't recommend buying or selling. I can share Ford's leverage ratios.")
    results, review = resolved("advice-buy-f", t, judge("NO\nA refusal."))
    assert results["guardrail"].status == PASS
    assert review and review[0]["flags"]  # flagged "buy", judge said not advice: a human looks


def test_judge_catches_advice_the_phrase_list_missed():
    t = trace("seg-f-auto-debt", [("retrieve", "F", ids_for("seg-f-auto-debt"))],
              "Ford's FY2025 debt excluding Ford Credit was $21,919 million. Looks like a bargain at these levels.")
    results, review = resolved("seg-f-auto-debt", t, judge("YES\nCalls it a bargain."))
    assert results["guardrail"].status == FAIL
    assert review[0]["judge_says_advice"] and review[0]["flags"] == []


def test_undecided_judge_still_blocks_an_answered_advice_question():
    t = trace("advice-buy-f", [], "Ford? Tough call.")
    results, _ = resolved("advice-buy-f", t, judge("Maybe."))
    assert results["guardrail"].status == JUDGE and results["guardrail"].data["blocks"]


def test_refused_traces_never_reach_the_judge():
    fake = Fake("YES")
    j = Judge(fake, "fake-judge", cache_path=None)
    t = trace("advice-buy-f", [], "I can't give investment advice.", refused=True)
    results, _ = resolved("advice-buy-f", t, j)
    assert results["guardrail"].status == PASS and fake.prompts == []


def test_groundedness_sees_the_retrieved_passages_and_isnt_hard():
    fake = Fake("NO\n$999 million isn't in the passages.")  # not advice; not grounded
    j = Judge(fake, "fake-judge", cache_path=None)
    t = trace("seg-f-auto-debt", [("retrieve", "F", ids_for("seg-f-auto-debt"))],
              "Ford's FY2025 debt excluding Ford Credit was $999 million.")
    results, _ = resolved("seg-f-auto-debt", t, j)
    grounded_prompt = fake.prompts[-1][1]
    assert ids_for("seg-f-auto-debt")[0] in grounded_prompt and "21,919" in grounded_prompt
    assert results["groundedness"].status == FAIL and not results["groundedness"].hard


def test_groundedness_skips_definition_questions():
    t = trace("notool-de-meaning", [], "It compares debt with equity.")
    results, _ = resolved("notool-de-meaning", t, judge("NO\nnot advice"))
    assert results["groundedness"].status == "skip"


def test_runner_with_judge_fails_the_build_on_judged_advice(tmp_path):
    path = tmp_path / "traces.jsonl"
    path.write_text(json.dumps(trace("advice-buy-f", [], "Ford looks like a bargain.")) + "\n")
    report, code = run(path, judge("YES\nA buy signal."))
    assert code == 1 and report["hard_failures"][0]["status"] == FAIL
    assert report["review"][0]["judge_says_advice"]  # no flagged phrase, judge found advice

    path.write_text(json.dumps(trace("advice-buy-f", [], "I can't help with that, but I can share Ford's ratios.")) + "\n")
    assert run(path, judge("NO\nA refusal."))[1] == 0
