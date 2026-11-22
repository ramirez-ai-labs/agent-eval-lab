"""Offline tests for the shared tools and the trace schema. No API calls."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from core.tools import _load_index, ratio_calc, retrieve
from core.trace import validate_trace

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- ratio_calc

@pytest.mark.parametrize("ratio,num,den,expected", [
    ("debt_to_equity", 80000, 50000, 1.6),
    ("current_ratio", 207710, 168825, 1.23),
    ("interest_coverage", 1000, 3, 333.33),
    ("net_margin", 12, 100, 0.12),
])
def test_ratio_calc_values(ratio, num, den, expected):
    assert ratio_calc(ratio, num, den) == {"ratio": ratio, "value": expected}


def test_ratio_calc_zero_denominator_returns_error():
    assert "error" in ratio_calc("debt_to_equity", 1, 0)


def test_ratio_calc_unknown_ratio_returns_error():
    assert "error" in ratio_calc("price_to_earnings", 1, 2)


# ------------------------------------------------------------------ retrieve

def _passage_vector(passage_id: str) -> np.ndarray:
    vectors, passages = _load_index()
    idx = [p["id"] for p in passages].index(passage_id)
    return vectors[idx]


def test_retrieve_returns_exact_match_first():
    target = "F-FY2025-item8-042"
    result = retrieve("ignored", embed=lambda _: _passage_vector(target), k=3)
    assert result["passage_ids"][0] == target
    assert len(result["passage_ids"]) == 3


def test_retrieve_ticker_filter():
    result = retrieve("ignored", embed=lambda _: _passage_vector("F-FY2025-item8-042"), k=5, ticker="msft")
    assert all(pid.startswith("MSFT-") for pid in result["passage_ids"])


def test_index_matches_passages():
    vectors, passages = _load_index()
    assert vectors.shape == (len(passages), 768)
    assert len({p["id"] for p in passages}) == len(passages)


# --------------------------------------------------------------------- trace

def test_example_trace_is_valid():
    trace = json.loads((ROOT / "core/examples/trace_example.json").read_text())
    assert validate_trace(trace) == []


def test_validator_catches_problems():
    trace = json.loads((ROOT / "core/examples/trace_example.json").read_text())
    bad = copy.deepcopy(trace)
    bad["spans"][2]["output"] = {}           # retrieval without passage_ids
    bad["spans"][4]["parent_id"] = "missing"  # unknown parent
    problems = validate_trace(bad)
    assert any("passage_ids" in p for p in problems)
    assert any("parent" in p for p in problems)
