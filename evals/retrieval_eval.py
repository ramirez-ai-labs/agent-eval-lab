"""Compare retrieval modes on the labeled retrieval set.

Metrics, averaged over queries:
  Recall@k  share of a query's relevant passages that appear in the top k
  Hit@k     share of queries with at least one relevant passage in the top k
  MRR       1 / rank of the first relevant passage (0 if none in the top 10)

Query embeddings are cached in evals/cache/, so reruns are free and work
offline. Only new queries call Workers AI.

Usage:
    uv run python evals/retrieval_eval.py
    uv run python evals/retrieval_eval.py --k 3 --show-misses
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.tools import retrieve, workers_ai_embedder  # noqa: E402
from evals.labels import resolve  # noqa: E402

DATASET = ROOT / "evals" / "datasets" / "retrieval.jsonl"
CACHE = ROOT / "evals" / "cache" / "query_embeddings.json"
MODES = ("vector", "bm25", "hybrid")
MRR_DEPTH = 10


def cached_embedder(model: str):
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    live = None

    def embed(query: str) -> np.ndarray:
        nonlocal live
        key = hashlib.sha256(f"{model}\n{query}".encode()).hexdigest()[:16]
        if key not in cache:
            if live is None:
                live = workers_ai_embedder(model)
            cache[key] = [round(float(x), 6) for x in live(query)]
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            CACHE.write_text(json.dumps(cache, indent=0) + "\n")
        return np.asarray(cache[key], dtype=np.float32)

    return embed


def score(ranked: list[str], relevant: set[str], k: int) -> dict[str, float]:
    top = ranked[:k]
    found = relevant & set(top)
    first = next((i for i, pid in enumerate(ranked[:MRR_DEPTH], 1) if pid in relevant), None)
    return {
        "recall": len(found) / len(relevant),
        "hit": 1.0 if found else 0.0,
        "mrr": 1.0 / first if first else 0.0,
        "rank": first,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--show-misses", action="store_true")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    model = yaml.safe_load(open(ROOT / "config" / "models.yaml"))["embeddings"]["model"]
    embed = cached_embedder(model)
    cases = [json.loads(line) for line in open(DATASET, encoding="utf-8")]

    results = {m: [] for m in MODES}
    for case in cases:
        for mode in MODES:
            ranked = retrieve(case["query"], embed, k=MRR_DEPTH, ticker=case["ticker"], mode=mode)["passage_ids"]
            relevant = set(resolve(case["evidence"], case["ticker"]))
            results[mode].append(score(ranked, relevant, args.k))

    print(f"Retrieval eval: {len(cases)} queries, k={args.k}\n")
    print(f"{'mode':<8} {'Recall@' + str(args.k):>10} {'Hit@' + str(args.k):>8} {'MRR':>6}")
    for mode in MODES:
        rs = results[mode]
        avg = {m: sum(r[m] for r in rs) / len(rs) for m in ("recall", "hit", "mrr")}
        print(f"{mode:<8} {avg['recall']:>10.2f} {avg['hit']:>8.2f} {avg['mrr']:>6.2f}")

    print(f"\nRank of first relevant passage (– = not in top {MRR_DEPTH}):")
    print(f"{'query':<22} " + " ".join(f"{m:>7}" for m in MODES))
    for i, case in enumerate(cases):
        cells = [str(results[m][i]["rank"] or "–") for m in MODES]
        print(f"{case['id']:<22} " + " ".join(f"{c:>7}" for c in cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
