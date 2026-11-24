"""Checks on the eval dataset itself, before any agent runs.

A wrong label is worse than no label: it fails a correct agent and passes a
wrong one. So every reference number is checked against the filings here,
including which column (fiscal year) it came from.
"""

import json
from collections import Counter
from pathlib import Path

import pytest

from core.tools import ratio_calc
from evals.labels import resolve

ROOT = Path(__file__).resolve().parents[1]
CASES = [json.loads(line) for line in open(ROOT / "evals" / "datasets" / "cases.jsonl", encoding="utf-8")]
CALIBRATION = [json.loads(line) for line in
               open(ROOT / "evals" / "datasets" / "judge_calibration.jsonl", encoding="utf-8")]

# Target counts from scenario/TASK.md §8.
TARGETS = {"fact": 6, "ratio": 6, "comparison": 4, "segment": 3, "fiscal_year": 2, "risk": 4,
           "no_tool": 3, "advice": 4, "adversarial": 2, "out_of_scope": 2}

# The latest fiscal year in each filing, and which filings list the oldest year first.
LATEST = {"MSFT": 2026, "NKE": 2026, "DUK": 2025, "F": 2025}
OLDEST_FIRST = {"F"}  # Ford's tables run 2023 → 2025; the others run newest → oldest

RATIOS = {"debt_to_equity", "current_ratio", "interest_coverage", "net_margin"}


def by_id(case):
    return case["id"]


def ratio_calls(case):
    return [t for t in case["expected_tools"] if t["name"] == "ratio_calc"]


def cells(row: str) -> list[str]:
    """The numeric cells of a 'Label | 1,234 | ( 56 )' row, without parentheses or spaces."""
    return [c.replace("(", "").replace(")", "").replace(" ", "") for c in row.split("|")[1:]]


def test_ids_are_unique():
    ids = [c["id"] for c in CASES]
    assert len(ids) == len(set(ids))


def test_category_counts_match_the_task_spec():
    assert Counter(c["category"] for c in CASES) == TARGETS


@pytest.mark.parametrize("case", CASES, ids=by_id)
def test_case_shape(case):
    assert case["question"].strip()
    assert case["reference_answer"].strip()
    assert set(case["tickers"]) <= set(LATEST)
    for tool in case["expected_tools"]:
        assert tool["name"] in ("retrieve", "ratio_calc")
    if case["expected_refusal"]:
        assert case["expected_tools"] == [] and case["expected_handoff"] is None
    assert (case["expected_handoff"] == "risk_agent") == (case["category"] == "risk")


@pytest.mark.parametrize("case", [c for c in CASES if c["evidence"] or c["evidence_any"]], ids=by_id)
def test_evidence_is_in_the_filings(case):
    for item in case["evidence"]:  # every one must be found
        resolve([item["text"]], item["ticker"])
        for alt in item.get("or", []):  # and so must each alternative
            resolve([alt["text"]], alt["ticker"])
    if case["evidence_any"]:  # alternatives: each must exist, any one is enough at eval time
        for item in case["evidence_any"]:
            resolve([item["text"]], item["ticker"])


@pytest.mark.parametrize("case", [c for c in CASES if ratio_calls(c)], ids=by_id)
def test_reference_ratios_recompute(case):
    for call in ratio_calls(case):
        args = call["args"]
        assert args["ratio"] in RATIOS
        assert ratio_calc(**args) == {"ratio": args["ratio"], "value": call["value"]}
        for side in ("numerator", "denominator"):
            parts = call.get(f"{side}_parts")
            if parts:
                assert sum(parts) == args[side], f"{side} parts don't add up"


@pytest.mark.parametrize("case", [c for c in CASES if ratio_calls(c)], ids=by_id)
def test_ratio_inputs_come_from_the_right_fiscal_year_column(case):
    """Each input appears in an evidence row, in the column for the case's fiscal year.

    This is the check that catches reading Ford's balance sheet left to right.
    """
    for call in ratio_calls(case):
        for side in ("numerator", "denominator"):
            for number in call.get(f"{side}_parts") or [call["args"][side]]:
                text = f"{number:,}"
                placed = []
                for item in case["evidence"]:
                    row = cells(item["text"])
                    if text not in row:
                        continue
                    ticker = item["ticker"]
                    year = int(case["fiscal_year"][2:]) if case["fiscal_year"] else LATEST[ticker]
                    back = LATEST[ticker] - year
                    col = len(row) - 1 - back if ticker in OLDEST_FIRST else back
                    placed.append(row.index(text) == col)
                assert placed, f"{text} isn't in any evidence row"
                assert any(placed), f"{text} is in the wrong fiscal-year column"


def test_judge_calibration_has_both_verdicts_and_both_traps():
    verdicts = Counter(c["is_advice"] for c in CALIBRATION)
    assert verdicts[True] and verdicts[False]
    kinds = {c["kind"] for c in CALIBRATION}
    assert {"phrase_false_fail", "phrase_false_pass"} <= kinds
    assert len({c["id"] for c in CALIBRATION}) == len(CALIBRATION)
