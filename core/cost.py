"""Measured cost: Workers AI Neurons, read from each API response.

Workers AI's Chat Completions responses include `usage.neurons`, the exact
cost of that call against the 10,000-Neuron free daily allowance. The lab
records it instead of estimating: the first live run (Nov 23, 2026) cost
about 6,500 Neurons for 5 cases, against an estimate of about 1,000.

How it fits together:
  - a run sets a Meter as the current one (`metering(meter)`)
  - the Workers AI client calls `record()` after each response
  - the trace recorder calls `meter.take()` to attach what was spent since
    its last take to the model call or guardrail span it is writing

Providers that don't report Neurons (OpenAI) record nothing, and traces
from them simply have no Neuron figures.

Not metered: query embeddings (bge-base-en-v1.5). They cost about 0.003
Neurons each, under 4 Neurons for a whole day of runs.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any


class Meter:
    """Neurons spent, in total and since the last take()."""

    def __init__(self) -> None:
        self.total = 0.0
        self.calls = 0
        self._pending: float | None = None

    def add(self, neurons: float) -> None:
        self.total += neurons
        self.calls += 1
        self._pending = (self._pending or 0.0) + neurons

    def take(self) -> float | None:
        """Neurons since the last take, or None if no call reported any."""
        pending, self._pending = self._pending, None
        return None if pending is None else round(pending, 2)


_CURRENT: ContextVar[Meter | None] = ContextVar("neuron_meter", default=None)


@contextmanager
def metering(meter: Meter) -> Iterator[Meter]:
    """Make `meter` the one that `record()` adds to, for the duration of a run."""
    token = _CURRENT.set(meter)
    try:
        yield meter
    finally:
        _CURRENT.reset(token)


def neurons_in(body: Any) -> float | None:
    """`usage.neurons` from a Workers AI response body, if it has one."""
    if not isinstance(body, dict):
        return None
    usage = body.get("usage")
    value = usage.get("neurons") if isinstance(usage, dict) else None
    return float(value) if isinstance(value, (int, float)) else None


def record(neurons: float | None) -> None:
    """Add to the current meter. Does nothing outside a metered run."""
    meter = _CURRENT.get()
    if meter is not None and neurons is not None:
        meter.add(neurons)
