"""Passage context from the extractor, offline. No downloads or API calls.

Context lines tell the agent where a passage came from. They aren't embedded,
so improving them is free: no re-embed (core.tools.texts_digest).
"""

import json
from pathlib import Path

from core.tools import texts_digest
from scenario.extract import context_line, split_passages

ROOT = Path(__file__).resolve().parents[1]
PASSAGES = {p["id"]: p for p in (json.loads(line) for line in
                                 open(ROOT / "scenario" / "filings" / "passages.jsonl", encoding="utf-8"))}

DEBT_NOTE = """7. DEBT AND CREDIT FACILITIES
The following tables summarize outstanding debt.
December 31, 2025
Notes payable and commercial paper | 3.95 | 3,254
Total debt | 4.41 | 89,836
December 31, 2024
Notes payable and commercial paper | 4.67 | 4,213
Total debt | 4.37 | 84,273
8. LEASES
Lease costs | 322 | 50"""


def periods_of(chunks, needle):
    return next(periods for text, _reg, _heads, periods in chunks if needle in text)


def test_each_year_table_is_labeled_with_its_year():
    """The Nov 24 failure: two debt tables that differ only by year."""
    # Whatever the chunk boundaries, a total's own year is always named, and the
    # 2024 total is never labeled 2025. (A chunk that ends with the next table's
    # date line names both years, which is accurate.)
    for size in (60, 90, 100, 120):
        chunks = split_passages(DEBT_NOTE, size=size, year_end="December 31")
        assert "December 31, 2025" in periods_of(chunks, "89,836")
        assert periods_of(chunks, "84,273") == ["December 31, 2024"]


def test_a_passage_spanning_both_tables_names_both_years():
    chunks = split_passages(DEBT_NOTE, size=2000, year_end="December 31")
    assert periods_of(chunks, "89,836") == ["December 31, 2025", "December 31, 2024"]


def test_a_new_heading_ends_the_period():
    chunks = split_passages(DEBT_NOTE + "\nLease income | 7 | 8", size=60, year_end="December 31")
    assert periods_of(chunks, "Lease income") == []


def test_only_fiscal_year_end_dates_count():
    """Report and signature dates sit on their own line too, but don't label tables."""
    text = "We have served as auditor since 1960.\nFebruary 26, 2026\nTotal | 1 | 2"
    assert split_passages(text, year_end="December 31")[0][3] == []
    assert split_passages("June 30, 2026\nRevenue | 1", year_end="June 30")[0][3] == ["June 30, 2026"]
    assert split_passages("December 31, 2025\nTotal | 1")[0][3] == []  # no year_end, no periods


def test_context_line_names_the_table_year():
    line = context_line("DUK", "FY2025", "item8", None, ["7. DEBT AND CREDIT FACILITIES"], ["December 31, 2024"])
    assert line.endswith("· table for December 31, 2024")


def test_duke_debt_tables_in_the_committed_passages():
    assert PASSAGES["DUK-FY2025-item8-254"]["context"].endswith("table for December 31, 2025")
    assert PASSAGES["DUK-FY2025-item8-256"]["context"].endswith("table for December 31, 2024")


def test_context_changes_dont_need_a_re_embed_but_text_changes_do():
    passages = [{"id": "a", "text": "Total debt | 89,836", "context": "Duke Energy"}]
    digest = texts_digest(passages)
    assert texts_digest([dict(passages[0], context="Duke Energy · table for December 31, 2025")]) == digest
    assert texts_digest([dict(passages[0], text="Total debt | 84,273")]) != digest
