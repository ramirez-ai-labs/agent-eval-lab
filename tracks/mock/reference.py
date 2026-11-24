"""The committed reference run that the mock track replays.

    runs/reference/<provider>/traces.jsonl   one trace per case, the latest recorded
    runs/reference/<provider>/manifest.json  where each trace came from

Live runs write to runs/<track>/<provider>/<stamp>/ (not committed). Promoting
copies a run's traces into the reference, replacing older traces for the same
cases, so a reference can be built up over several days of free allowance:

    uv run python -m tracks.mock.reference runs/openai/workers-ai/<stamp>/traces.jsonl
    uv run python -m tracks.mock.reference --status     # which cases are still missing
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.trace import load_traces, validate_trace  # noqa: E402

REFERENCE = ROOT / "runs" / "reference"
CASES = ROOT / "evals" / "datasets" / "cases.jsonl"


def paths(provider: str = "workers-ai", root: Path = REFERENCE) -> tuple[Path, Path]:
    folder = root / provider
    return folder / "traces.jsonl", folder / "manifest.json"


def dataset_ids() -> list[str]:
    return [json.loads(line)["id"] for line in open(CASES, encoding="utf-8")]


def load(provider: str = "workers-ai", root: Path = REFERENCE) -> tuple[list[dict], dict]:
    """(traces in dataset order, manifest). Empty if there's no reference yet."""
    traces_path, manifest_path = paths(provider, root)
    if not traces_path.exists():
        return [], {"provider": provider, "cases": {}}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return load_traces(traces_path), manifest


def missing(provider: str = "workers-ai", root: Path = REFERENCE) -> list[str]:
    """Dataset cases with no reference trace yet, in dataset order."""
    have = {t["case_id"] for t in load(provider, root)[0]}
    return [i for i in dataset_ids() if i not in have]


def promote(run_paths: list[Path], provider: str = "workers-ai", root: Path = REFERENCE) -> list[str]:
    """Copy traces from live runs into the reference. Later runs win per case.

    Returns the case IDs promoted. Only dataset cases with valid traces are
    taken; ad-hoc questions and planted traces are refused.
    """
    known = set(dataset_ids())
    traces, manifest = load(provider, root)
    by_case = {t["case_id"]: t for t in traces}
    promoted = []
    for path in run_paths:
        for t in load_traces(path):
            if t["case_id"] not in known:
                continue
            if t.get("planted") or t["provider"] == "mock":
                raise ValueError(f"{path}: {t['case_id']} is a mock trace; only live runs can be promoted")
            if t["provider"] != provider:
                raise ValueError(f"{path}: {t['case_id']} is from {t['provider']}, not {provider}")
            problems = validate_trace(t)
            if problems:
                raise ValueError(f"{path}: invalid trace for {t['case_id']}: {problems}")
            by_case[t["case_id"]] = t
            manifest["cases"][t["case_id"]] = {
                "source": str(Path(path).resolve().relative_to(ROOT)) if Path(path).resolve().is_relative_to(ROOT)
                else str(path),
                "model": t["model"],
                "neurons": t.get("neurons"),
            }
            promoted.append(t["case_id"])

    order = {cid: i for i, cid in enumerate(dataset_ids())}
    ordered = sorted(by_case.values(), key=lambda t: order[t["case_id"]])
    manifest["cases"] = {cid: manifest["cases"][cid] for cid in sorted(manifest["cases"], key=order.get)}
    manifest["updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    manifest["complete"] = len(ordered) == len(order)

    traces_path, manifest_path = paths(provider, root)
    traces_path.parent.mkdir(parents=True, exist_ok=True)
    traces_path.write_text("".join(json.dumps(t) + "\n" for t in ordered), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return promoted


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the reference run the mock track replays.")
    parser.add_argument("runs", nargs="*", type=Path, help="traces.jsonl files from live runs, oldest first")
    parser.add_argument("--provider", default="workers-ai")
    parser.add_argument("--status", action="store_true", help="show which dataset cases are still missing")
    args = parser.parse_args()
    if args.runs:
        promoted = promote(args.runs, args.provider)
        print(f"promoted {len(promoted)} trace(s): {', '.join(promoted) or '-'}")
    gaps = missing(args.provider)
    total = len(dataset_ids())
    print(f"reference ({args.provider}): {total - len(gaps)}/{total} cases")
    if gaps and (args.status or args.runs):
        print("missing: " + ", ".join(gaps))
    return 0


if __name__ == "__main__":
    sys.exit(main())
