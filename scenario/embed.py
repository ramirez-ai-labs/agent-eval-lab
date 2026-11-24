"""Embed every passage once with Workers AI and commit the vectors.

Output (committed):
    scenario/index/vectors.npy   float16, shape (n_passages, 768), L2-normalized
    scenario/index/meta.json     model, dimension, passage IDs in row order,
                                 and a hash of each passage's ID and text

Retrieval loads these files and only embeds the query at run time, so
nobody has to re-embed the filings. `retrieve` refuses to run if
any passage's ID or text changed since the vectors were built. Context lines
aren't embedded, so changing only those doesn't need a re-embed.

Usage:
    uv run python scenario/embed.py
    uv run python scenario/embed.py --check   # does the index still match? (free)

Cost: about 3,300 Neurons of the 10,000 free daily allowance (measured
Nov 22, 2026), so at most one full re-embed a day.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import requests
import yaml
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from core.tools import texts_digest  # noqa: E402
PASSAGES = HERE / "filings" / "passages.jsonl"
INDEX = HERE / "index"
BATCH = 50


def embed_texts(texts: list[str], model: str, account_id: str, token: str) -> np.ndarray:
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
    r = requests.post(url, headers={"Authorization": f"Bearer {token}"}, json={"text": texts}, timeout=120)
    r.raise_for_status()
    return np.asarray(r.json()["result"]["data"], dtype=np.float32)


def check() -> int:
    """Report whether the committed vectors still match the passages. No API calls."""
    meta = json.loads((INDEX / "meta.json").read_text(encoding="utf-8"))
    passages = [json.loads(line) for line in open(PASSAGES, encoding="utf-8")]
    if texts_digest(passages) == meta.get("texts_sha256"):
        print(f"index matches: {len(passages)} passages, IDs and text unchanged. No re-embed needed.")
        return 0
    print("passage IDs or text changed since the vectors were built. Run: uv run python scenario/embed.py")
    return 1


def main() -> int:
    if "--check" in sys.argv[1:]:
        return check()
    load_dotenv(ROOT / ".env")
    account_id, token = os.getenv("CF_ACCOUNT_ID"), os.getenv("CF_API_TOKEN")
    if not account_id or not token:
        print("Missing CF_ACCOUNT_ID or CF_API_TOKEN. Copy .env.example to .env and fill them in.")
        return 2

    model = yaml.safe_load(open(ROOT / "config" / "models.yaml"))["embeddings"]["model"]
    passages = [json.loads(line) for line in open(PASSAGES, encoding="utf-8")]

    chunks = []
    for i in range(0, len(passages), BATCH):
        batch = passages[i : i + BATCH]
        # Text only. Embedding the context line too dropped vector Recall@5
        # from 0.55 to 0.26: the shared prefix made hundreds of vectors look
        # alike. See evals/README.md.
        chunks.append(embed_texts([p["text"] for p in batch], model, account_id, token))
        print(f"  embedded {min(i + BATCH, len(passages))}/{len(passages)}", end="\r")
        time.sleep(0.1)
    vectors = np.vstack(chunks)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)  # cosine = dot product

    INDEX.mkdir(exist_ok=True)
    np.save(INDEX / "vectors.npy", vectors.astype(np.float16))
    meta = {
        "model": model,
        "dimension": int(vectors.shape[1]),
        "count": len(passages),
        "texts_sha256": texts_digest(passages),
        "ids": [p["id"] for p in passages],
    }
    (INDEX / "meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    print(f"\n{len(passages)} vectors ({vectors.shape[1]} dims) → {INDEX}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
