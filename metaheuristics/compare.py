"""Benchmark several optimizers on a function and tabulate the outcome.

A tiny harness for apples-to-apples comparisons: same objective, same box, same
evaluation budget, multiple seeds. It returns plain dicts so it stays
dependency-free; pretty-print with :func:`format_table` or feed the numbers to
your own plotting code.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .core import Bounds, Optimizer


def run_trials(
    optimizer: Optimizer,
    func,
    bounds: Bounds,
    max_iter: int = 200,
    seeds: Sequence[int] = range(10),
) -> dict:
    """Run ``optimizer`` over several seeds and summarize best-value statistics."""
    finals, evals, curves = [], [], []
    for s in seeds:
        res = optimizer.minimize(func, bounds, max_iter=max_iter, seed=s)
        finals.append(res.best_f)
        evals.append(res.n_evals)
        curves.append(res.history)
    finals = np.asarray(finals)
    return {
        "name": optimizer.name,
        "mean": float(finals.mean()),
        "std": float(finals.std()),
        "best": float(finals.min()),
        "worst": float(finals.max()),
        "median": float(np.median(finals)),
        "evals": int(np.mean(evals)),
        "curves": curves,
    }


def compare(
    optimizers: Sequence[Optimizer],
    func,
    bounds: Bounds,
    max_iter: int = 200,
    seeds: Sequence[int] = range(10),
) -> list[dict]:
    """Benchmark a list of optimizers, sorted best-mean-first."""
    rows = [run_trials(o, func, bounds, max_iter, seeds) for o in optimizers]
    return sorted(rows, key=lambda r: r["mean"])


def format_table(rows: list[dict], f_min: Optional[float] = None) -> str:
    """Render :func:`compare` rows as a monospace table."""
    head = f"{'optimizer':<16} {'mean':>12} {'std':>11} {'best':>12} {'evals':>8}"
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{r['name']:<16} {r['mean']:>12.4g} {r['std']:>11.4g} "
            f"{r['best']:>12.4g} {r['evals']:>8d}"
        )
    if f_min is not None:
        lines.append(f"\n(global minimum f* = {f_min:g})")
    return "\n".join(lines)
