"""Grade a traces.jsonl file against the dataset with the code checks.

Usage:
    uv run python -m evals.run_checks runs/openai/workers-ai/<stamp>/traces.jsonl

Prints one row per case and a pass rate per check, and writes report.json
next to the traces. Exit code 1 means a hard check failed: any guardrail
failure fails the build, because safety failures should never be averaged away.

Checks marked "judge" need the LLM judge, which isn't wired in yet; they're
listed, not counted. One kind still blocks the build: an advice question that
got an answer instead of a refusal. Until the judge rules, it isn't known to be safe.
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
         "numeric_accuracy": "numbers", "handoff": "handoff", "guardrail": "guardrail",
         "fiscal_year": "fiscal yr", "budget": "budget"}


def load_cases() -> dict[str, dict]:
    path = ROOT / "evals" / "datasets" / "cases.jsonl"
    return {c["id"]: c for c in (json.loads(line) for line in open(path, encoding="utf-8"))}


def run(traces_path: Path) -> tuple[dict, int]:
    cases = load_cases()
    with open(ROOT / "config" / "models.yaml", encoding="utf-8") as f:
        budgets = yaml.safe_load(f)["budgets"]

    graded, unknown = {}, []
    for trace in load_traces(traces_path):
        case = cases.get(trace["case_id"])
        if case is None:
            unknown.append(trace["case_id"])
            continue
        graded[trace["case_id"]] = grade(case, trace, budgets)

    totals: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for results in graded.values():
        for r in results:
            totals[r.check][r.status] += 1
    hard_failures = [(cid, r) for cid, results in graded.items() for r in results
                     if r.hard and (r.status == FAIL or (r.status == JUDGE and r.data.get("blocks")))]

    report = {
        "traces": str(traces_path),
        "cases_graded": len(graded),
        "not_in_dataset": unknown,
        "pass_rates": {check: {"pass": t[PASS], "fail": t[FAIL], "judge": t[JUDGE],
                               "rate": round(t[PASS] / (t[PASS] + t[FAIL]), 3) if t[PASS] + t[FAIL] else None}
                       for check, t in totals.items()},
        "hard_failures": [{"case": cid, "check": r.check, "status": r.status, "detail": r.detail}
                          for cid, r in hard_failures],
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
    args = parser.parse_args()
    report, code = run(args.traces)
    print_report(report)
    out = args.traces.parent / "report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport → {out}")
    return code


if __name__ == "__main__":
    sys.exit(main())
