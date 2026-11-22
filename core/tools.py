"""Shared tools: the same retrieval and math for every track.

These are plain Python functions with no agent framework imports. Each track
wraps them in its own SDK's tool format, so every track retrieves the same
passages and computes the same ratios, and only agent behavior varies.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
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


_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class _BM25:
    """Okapi BM25 keyword search, built in memory from the passages (~0.5 s)."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tfs = [Counter(_tokens(d)) for d in docs]
        self.lengths = np.array([sum(tf.values()) for tf in self.tfs], dtype=np.float32)
        self.avg_len = float(self.lengths.mean())
        df = Counter(term for tf in self.tfs for term in tf)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.tfs), dtype=np.float32)
        norm = self.k1 * (1 - self.b + self.b * self.lengths / self.avg_len)
        for term in set(_tokens(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            tf = np.array([d.get(term, 0) for d in self.tfs], dtype=np.float32)
            out += idf * tf * (self.k1 + 1) / (tf + norm)
        return out


@lru_cache(maxsize=1)
def _load_bm25() -> _BM25:
    _, passages = _load_index()
    return _BM25([p["text"] for p in passages])


RRF_K = 60  # standard reciprocal rank fusion constant


def _rank(scores: np.ndarray, candidates: np.ndarray) -> list[int]:
    return [int(i) for i in candidates[np.argsort(-scores[candidates], kind="stable")]]


def retrieve(
    query: str,
    embed: Embedder | None = None,
    k: int = 5,
    ticker: str | None = None,
    mode: Literal["hybrid", "vector", "bm25"] = "hybrid",
) -> dict:
    """Return the top-k passages for a query, optionally for one company.

    Modes:
      vector  semantic search: good at meaning, weak on number-heavy tables
      bm25    keyword search: good at exact terms like "Total current assets"
      hybrid  both, merged with reciprocal rank fusion (the default)

    The output always includes "passage_ids". The trace schema requires it,
    because the Recall@k and MRR checks read it.
    """
    vectors, passages = _load_index()
    if ticker:
        candidates = np.array([i for i, p in enumerate(passages) if p["ticker"] == ticker.upper()])
    else:
        candidates = np.arange(len(passages))
    if len(candidates) == 0:
        return {"passage_ids": [], "passages": []}

    rankings: list[list[int]] = []
    if mode in ("vector", "hybrid"):
        if embed is None:
            raise ValueError(f"mode '{mode}' needs an embed function")
        q = embed(query)
        rankings.append(_rank(vectors @ (q / np.linalg.norm(q)), candidates))
    if mode in ("bm25", "hybrid"):
        rankings.append(_rank(_load_bm25().scores(query), candidates))

    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, i in enumerate(ranking, 1):
            fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank)
    top = sorted(fused, key=lambda i: (-fused[i], i))[:k]

    hits = [{
        "id": passages[i]["id"],
        "ticker": passages[i]["ticker"],
        "fiscal_year": passages[i]["fiscal_year"],
        "section": passages[i]["section"],
        "score": round(fused[i], 5),
        "text": passages[i]["text"],
    } for i in top]
    return {"passage_ids": [h["id"] for h in hits], "passages": hits}
