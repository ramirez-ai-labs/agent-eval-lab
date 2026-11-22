"""Evidence-based labels: find which passages contain each case's evidence.

Labels store the exact text that answers a question ("evidence"), not
passage IDs. IDs are resolved from the evidence at eval time, so changing
how filings are chunked doesn't silently break the labels. If evidence can't
be found, for example because a chunk boundary split it, resolution fails
loudly instead.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PASSAGES = ROOT / "scenario" / "filings" / "passages.jsonl"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=1)
def _passages() -> list[tuple[str, str, str]]:
    rows = (json.loads(line) for line in open(PASSAGES, encoding="utf-8"))
    return [(p["id"], p["ticker"], _norm(p["text"])) for p in rows]


def resolve(evidence: list[str], ticker: str | None = None) -> list[str]:
    """Passage IDs that contain any of the evidence strings."""
    ids: list[str] = []
    for snippet in evidence:
        needle = _norm(snippet)
        found = [pid for pid, t, text in _passages() if needle in text and (ticker is None or t == ticker)]
        if not found:
            raise LookupError(f"evidence not found in any passage: {snippet!r}")
        ids.extend(pid for pid in found if pid not in ids)
    return ids
