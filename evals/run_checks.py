"""Grade a traces.jsonl file against the dataset with the code checks.

Usage:
    uv run python -m evals.run_checks runs/openai/workers-ai/<stamp>/traces.jsonl
    uv run python -m evals.run_checks <traces.jsonl> --judge    # also run the LLM judge
    uv run python -m evals.run_checks <traces.jsonl> --judge-cached   # judge from the cache only: offline

Prints one row per case and a pass rate per check, and writes report.json
next to the traces. Exit code 1 means a hard check failed: any guardrail
failure fails the build, because safety failures should never be averaged away.

Without --judge, checks marked "judge" are listed, not counted, and grading is
free and offline. One kind still blocks the build: an advice question that got
an answer instead of a refusal. Until the judge rules, it isn't known to be safe.

With --judge, the LLM judge decides those cases and checks groundedness, and
every case where the phrase flags and the judge disagree goes on a review list.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import yaml

from core.trace import load_traces
from evals.checks import FAIL, JUDGE, PASS, grade

ROOT = Path(__file__).resolve().parents[1]
MARK = {PASS: "✓", FAIL: "✗", JUDGE: "?", "skip": "·"}
LABEL = {"evidence_recall": "evidence", "tool_choice": "tools", "ratio_args": "ratio args",
         "ratio_first_try": "1st try", "numeric_accuracy": "numbers", "handoff": "handoff", "guardrail": "guardrail",
         "fiscal_year": "fiscal yr", "budget": "budget", "groundedness": "grounded"}


def load_cases() -> dict[str, dict]:
    path = ROOT / "evals" / "datasets" / "cases.jsonl"
    return {c["id"]: c for c in (json.loads(line) for line in open(path, encoding="utf-8"))}


def run(traces_path: Path, judge=None) -> tuple[dict, int]:
    cases = load_cases()
    with open(ROOT / "config" / "models.yaml", encoding="utf-8") as f:
        budgets = yaml.safe_load(f)["budgets"]

    graded, unknown, review, neurons = {}, [], [], {}
    for trace in load_traces(traces_path):
        if "neurons" in trace:
            neurons[trace["case_id"]] = trace["neurons"]
        case = cases.get(trace["case_id"])
        if case is None:
            unknown.append(trace["case_id"])
            continue
        results = grade(case, trace, budgets)
        if judge is not None:
            from evals.judge import apply
            results, disagreements = apply(case, trace, results, judge)
            review.extend(disagreements)
        graded[trace["case_id"]] = results

    totals: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for results in graded.values():
        for r in results:
            totals[r.check][r.status] += 1
    hard_failures = [(cid, r) for cid, results in graded.items() for r in results
                     if r.hard and (r.status == FAIL or (r.status == JUDGE and r.data.get("blocks")))]

    report = {
        "traces": str(traces_path),
        "judge": (f"{judge.model} [{judge.variant}]" if getattr(judge, "variant", "") else judge.model)
                 if judge is not None else None,
        "cases_graded": len(graded),
        "not_in_dataset": unknown,
        "pass_rates": {check: {"pass": t[PASS], "fail": t[FAIL], "judge": t[JUDGE],
                               "rate": round(t[PASS] / (t[PASS] + t[FAIL]), 3) if t[PASS] + t[FAIL] else None}
                       for check, t in totals.items()},
        "hard_failures": [{"case": cid, "check": r.check, "status": r.status, "detail": r.detail}
                          for cid, r in hard_failures],
        "review": review,
        "neurons": {
            "agent": round(sum(neurons.values()), 2) if neurons else None,
            "agent_per_case": neurons,
            "judge": round(judge.meter.total, 2) if getattr(judge, "meter", None) and judge.meter.calls else None,
            "judge_new_calls": getattr(judge, "calls", 0),
        },
        "cases": {cid: [asdict(r) for r in results] for cid, results in graded.items()},
    }
    return report, (1 if hard_failures else 0)


def print_report(report: dict) -> None:
    checks = list(report["pass_rates"])
    print(f"{'case':<26}" + "".join(f"{LABEL.get(c, c):>11}" for c in checks))
    for cid, results in report["cases"].items():
        status = {r["check"]: r["status"] for r in results}
        print(f"{cid:<26}" + "".join(f"{MARK[status[c]]:>11}" for c in checks))
    print("\npass rate (judge cases not counted)")
    for check, t in report["pass_rates"].items():
        rate = "n/a" if t["rate"] is None else f"{t['rate']:.0%}"
        extra = f", {t['judge']} for the judge" if t["judge"] else ""
        print(f"  {check:<18} {rate:>5}  ({t['pass']} pass, {t['fail']} fail{extra})")
    print("\ndetails (✗ fail, ? for the judge)")
    for cid, results in report["cases"].items():
        for r in results:
            if r["status"] in (FAIL, JUDGE) and r["detail"]:
                print(f"  {MARK[r['status']]} {cid} · {r['check']}: {r['detail']}")
    n = report.get("neurons") or {}
    if n.get("agent") is not None or n.get("judge") is not None:
        print("\nNeurons (measured)")
        if n.get("agent") is not None:
            top = max(n["agent_per_case"].items(), key=lambda kv: kv[1])
            print(f"  agent  {n['agent']:>8,.0f}  over {len(n['agent_per_case'])} case(s); "
                  f"most: {top[0]} ({top[1]:,.0f})")
        if n.get("judge") is not None:
            print(f"  judge  {n['judge']:>8,.0f}  over {n['judge_new_calls']} new call(s); cached verdicts are free")
    if report["review"]:
        print("\nfor human review: the phrase flags and the judge disagree")
        for item in report["review"]:
            side = "judge found advice the phrase list missed" if item["judge_says_advice"] else \
                f"flagged {item['flags']}, judge says not advice"
            print(f"  {item['case']}: {side}. {item['reason'][:100]}")
    if report["not_in_dataset"]:
        print(f"\nnot in the dataset, not graded: {report['not_in_dataset']}")
    if report["hard_failures"]:
        fails = sum(f["status"] == FAIL for f in report["hard_failures"])
        pending = len(report["hard_failures"]) - fails
        print(f"\nHARD FAIL: {fails} guardrail failure(s), {pending} advice answer(s) awaiting the judge. "
              "This fails the build.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Grade traces with the code checks.")
    parser.add_argument("traces", type=Path)
    parser.add_argument("--judge", action="store_true", help="run the LLM judge (uses Workers AI; verdicts are cached)")
    parser.add_argument("--judge-cached", action="store_true",
                        help="use cached judge verdicts only: no API calls or credentials; misses stay undecided")
    args = parser.parse_args()
    judge = None
    if args.judge or args.judge_cached:
        from evals.judge import judge_from_config
        judge = judge_from_config(cached_only=args.judge_cached and not args.judge)
    try:
        report, code = run(args.traces, judge)
    except Exception as e:
        if judge is None:
            raise
        from evals.judge import explain
        print(f"Judge error {type(e).__name__}: {explain(e)}")
        print("Verdicts so far are cached; rerun to continue where it stopped.")
        return 2
    finally:
        if judge is not None:
            judge.save()
    print_report(report)
    if judge is not None:
        print(f"\njudge: {judge.model} · {judge.calls} new calls, the rest cached")
    out = args.traces.parent / "report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport → {out}")
    return code


if __name__ == "__main__":
    sys.exit(main())
