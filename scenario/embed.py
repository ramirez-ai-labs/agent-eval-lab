"""Embed every passage once with Workers AI and commit the vectors.

Output (committed):
    scenario/index/vectors.npy   float16, shape (n_passages, 768), L2-normalized
    scenario/index/meta.json     model, dimension, passage IDs in row order,
                                 and a hash of passages.jsonl

Retrieval loads these files and only embeds the query at run time, so
nobody has to re-embed the filings. `retrieve` refuses to run if
passages.jsonl changed since the vectors were built.

Usage:
    uv run python scenario/embed.py

Cost: about 3,000 Neurons of the 10,000 free daily allowance, once.
"""

from __future__ import annotations

import hashlib
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
PASSAGES = HERE / "filings" / "passages.jsonl"
INDEX = HERE / "index"
BATCH = 50


def passages_hash() -> str:
    return hashlib.sha256(PASSAGES.read_bytes()).hexdigest()


def embed_texts(texts: list[str], model: str, account_id: str, token: str) -> np.ndarray:
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
    r = requests.post(url, headers={"Authorization": f"Bearer {token}"}, json={"text": texts}, timeout=120)
    r.raise_for_status()
    return np.asarray(r.json()["result"]["data"], dtype=np.float32)


def main() -> int:
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
        "passages_sha256": passages_hash(),
        "ids": [p["id"] for p in passages],
    }
    (INDEX / "meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    print(f"\n{len(passages)} vectors ({vectors.shape[1]} dims) → {INDEX}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
