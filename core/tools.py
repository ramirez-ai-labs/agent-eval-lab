"""Shared tools: the same retrieval and math for every track.

These are plain Python functions with no agent framework imports. Each track
wraps them in its own SDK's tool format, so every track retrieves the same
passages and computes the same ratios, and only agent behavior varies.
"""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
PASSAGES = ROOT / "scenario" / "filings" / "passages.jsonl"
INDEX = ROOT / "scenario" / "index"

# ---------------------------------------------------------------- ratio_calc

Ratio = Literal["debt_to_equity", "current_ratio", "interest_coverage", "net_margin"]
RATIOS: dict[str, str] = {   # name -> formula, as pinned in scenario/TASK.md §4
    "debt_to_equity": "total debt / total shareholders' equity",
    "current_ratio": "total current assets / total current liabilities",
    "interest_coverage": "operating income / interest expense",
    "net_margin": "net income attributable to the company / total revenue",
}


def ratio_calc(ratio: Ratio, numerator: float, denominator: float) -> dict:
    """Compute a financial ratio with the formulas in scenario/TASK.md.

    Returns {"ratio", "value"} with value rounded to 2 decimals, or
    {"ratio", "error"} for an unknown ratio or a zero denominator. Errors are
    returned, not raised, so the agent sees them and can recover.
    """
    if ratio not in RATIOS:
        return {"ratio": ratio, "error": f"unknown ratio; use one of {sorted(RATIOS)}"}
    if denominator == 0:
        return {"ratio": ratio, "error": "denominator is zero"}
    return {"ratio": ratio, "value": round(numerator / denominator, 2)}


# ------------------------------------------------------------------ retrieve

Embedder = Callable[[str], np.ndarray]


@lru_cache(maxsize=1)
def _load_index() -> tuple[np.ndarray, list[dict]]:
    meta = json.loads((INDEX / "meta.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(PASSAGES.read_bytes()).hexdigest()
    if digest != meta["passages_sha256"]:
        raise RuntimeError(
            "passages.jsonl changed since the vectors were built. "
            "Run: uv run python scenario/embed.py"
        )
    vectors = np.load(INDEX / "vectors.npy").astype(np.float32)
    passages = [json.loads(line) for line in open(PASSAGES, encoding="utf-8")]
    if [p["id"] for p in passages] != meta["ids"]:
        raise RuntimeError("passage IDs don't match the index. Run: uv run python scenario/embed.py")
    return vectors, passages


def workers_ai_embedder(model: str = "@cf/baai/bge-base-en-v1.5") -> Embedder:
    """Embed a query with the same Workers AI model used for the passages."""
    account_id, token = os.environ["CF_ACCOUNT_ID"], os.environ["CF_API_TOKEN"]
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"

    def embed(query: str) -> np.ndarray:
        r = requests.post(url, headers={"Authorization": f"Bearer {token}"}, json={"text": [query]}, timeout=60)
        r.raise_for_status()
        return np.asarray(r.json()["result"]["data"][0], dtype=np.float32)

    return embed


def retrieve(
    query: str,
    embed: Embedder,
    k: int = 5,
    ticker: str | None = None,
) -> dict:
    """Return the top-k passages for a query, optionally for one company.

    The output always includes "passage_ids". The trace schema requires it,
    because the Recall@k and MRR checks read it.
    """
    vectors, passages = _load_index()
    q = embed(query)
    q = q / np.linalg.norm(q)
    scores = vectors @ q
    order = np.argsort(-scores)
    hits = []
    for i in order:
        p = passages[i]
        if ticker and p["ticker"] != ticker.upper():
            continue
        hits.append({
            "id": p["id"],
            "ticker": p["ticker"],
            "fiscal_year": p["fiscal_year"],
            "section": p["section"],
            "score": round(float(scores[i]), 4),
            "text": p["text"],
        })
        if len(hits) == k:
            break
    return {"passage_ids": [h["id"] for h in hits], "passages": hits}
