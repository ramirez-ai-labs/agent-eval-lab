"""The mock track: replay the committed reference run, offline and free.

No model, no credentials, no network. Every trace comes from a real recorded
run (runs/reference/), so the checks grade real agent behavior. Use it for:
  - CI: grading the reference on every pull request
  - the offline workshop tier, when there's no account or the allowance is used up
  - Lab 2: planted failures the checks must catch (tracks/mock/plants.py)

Usage:
    uv run python -m tracks.mock.track                         # replay the reference
    uv run python -m tracks.mock.track --plant all             # plus every planted failure
    uv run python -m tracks.mock.track --plant math-in-head --cases ratio-f-cr
    uv run python -m evals.run_checks runs/mock/<stamp>/traces.jsonl --judge-cached

A planted failure replaces the clean trace of its case. Grading a run with
`--plant advice-answered` fails the build (exit code 1), by design.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.trace import validate_trace, write_trace  # noqa: E402
from tracks.mock import reference  # noqa: E402
from tracks.mock.plants import PLANTS  # noqa: E402


def replay(provider: str = "workers-ai", plants: list[str] | None = None, cases: list[str] | None = None,
           root: Path = reference.REFERENCE) -> list[dict]:
    """Reference traces as mock traces, with the named failures planted."""
    traces, _ = reference.load(provider, root)
    if not traces:
        raise SystemExit(f"no reference run for {provider} yet. See tracks/mock/reference.py.")
    by_case = {t["case_id"]: t for t in traces}
    unknown = [p for p in plants or [] if p not in PLANTS]
    if unknown:
        raise SystemExit(f"unknown plant(s) {unknown}; choose from {sorted(PLANTS)}")
    out = {}
    for cid, t in by_case.items():
        out[cid] = {**t, "track": "mock", "provider": "mock", "model": f"replay of {t['model']}"}
    for name in plants or []:
        target, plant = PLANTS[name]
        if target not in out:
            raise SystemExit(f"plant {name} needs a reference trace for {target}")
        out[target] = {**plant(out[target]), "planted": name}
    selected = [out[c] for c in (cases or out) if c in out]
    for t in selected:
        problems = validate_trace(t)
        if problems:
            raise ValueError(f"invalid mock trace for {t['case_id']}: {problems}")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay the reference run offline.")
    parser.add_argument("--plant", default="", help=f"comma-separated, or 'all': {', '.join(PLANTS)}")
    parser.add_argument("--cases", help="comma-separated case IDs (default: every reference case)")
    parser.add_argument("--provider", default="workers-ai", help="which reference run to replay")
    parser.add_argument("--out", type=Path, help="traces.jsonl to write (default: runs/mock/<stamp>/)")
    args = parser.parse_args()

    plants = list(PLANTS) if args.plant == "all" else [p for p in args.plant.split(",") if p]
    traces = replay(args.provider, plants, args.cases.split(",") if args.cases else None)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or ROOT / "runs" / "mock" / stamp / "traces.jsonl"
    if out.exists():
        out.unlink()  # a rerun replaces the file, so replays stay deterministic
    for t in traces:
        write_trace(t, out)

    _, manifest = reference.load(args.provider)
    total = len(reference.dataset_ids())
    print(f"replayed {len(traces)} trace(s) from the {args.provider} reference "
          f"({len(manifest['cases'])}/{total} cases recorded)")
    for t in traces:
        if t.get("planted"):
            print(f"  planted {t['planted']} in {t['case_id']}")
    shown = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
    print(f"\nTraces → {shown}")
    print(f"Grade:   uv run python -m evals.run_checks {shown} --judge-cached")
    return 0


if __name__ == "__main__":
    sys.exit(main())
