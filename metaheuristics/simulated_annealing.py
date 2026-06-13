"""Simulated Annealing.

SA is a single-point search that imitates the cooling of a metal. From the
current point ``x`` it proposes a neighbour ``x'`` and always accepts an
improvement; a *worse* move is accepted with probability

    P = exp(-(f(x') - f(x)) / T)

The *temperature* ``T`` starts high (almost any move is accepted -- the search
roams freely and escapes local minima) and is lowered on a *cooling schedule*
toward zero (only improving moves survive -- the search settles). The art is in
the schedule: too fast and you freeze in a bad basin, too slow and you waste
evaluations.

This implementation uses geometric cooling ``T <- T * alpha`` with a Gaussian
proposal whose step size shrinks with the temperature, plus restart-to-best so a
long unlucky streak cannot drift away from the incumbent.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .core import Bounds, Optimizer


class SimulatedAnnealing(Optimizer):
    """Geometric-cooling SA with temperature-scaled Gaussian proposals.

    Parameters
    ----------
    t_start:
        Initial temperature. A good rule of thumb is to set it so that a typical
        worsening move is accepted ~80% of the time at the start.
    t_end:
        Final temperature (the schedule is calibrated to reach it by the last
        iteration).
    step_frac:
        Proposal std at ``t_start`` as a fraction of each axis span; it scales
        down with ``sqrt(T / t_start)``.
    restart_after:
        Reset the walker to the best-known point after this many non-improving
        steps. ``0`` disables it.
    """

    name = "SA"

    def __init__(
        self,
        t_start: float = 1.0,
        t_end: float = 1e-3,
        step_frac: float = 0.2,
        restart_after: int = 50,
    ):
        self.t_start = t_start
        self.t_end = t_end
        self.step_frac = step_frac
        self.restart_after = restart_after

    def _run(self, func, bounds: Bounds, max_iter, rng, callback):
        d = bounds.dim
        alpha = (self.t_end / self.t_start) ** (1.0 / max(1, max_iter - 1))

        x = bounds.sample(1, rng)[0]
        fx = func(x)
        best_x, best_f = x.copy(), fx
        history = [best_f]

        T = self.t_start
        since_improve = 0
        for it in range(max_iter):
            step = self.step_frac * bounds.span * np.sqrt(max(T / self.t_start, 1e-12))
            cand = bounds.reflect(x + rng.normal(0.0, step))
            fc = func(cand)
            delta = fc - fx
            if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-12)):
                x, fx = cand, fc

            if fx < best_f:
                best_f, best_x = fx, x.copy()
                since_improve = 0
            else:
                since_improve += 1

            if self.restart_after and since_improve >= self.restart_after:
                x, fx = best_x.copy(), best_f
                since_improve = 0

            T *= alpha
            history.append(best_f)
            self._emit(callback, best_x, best_f, history, it + 1, self.name)

        return best_x, best_f, history, False, {"t_end": T}


def sa(func, bounds: Bounds, max_iter: int = 500, seed: Optional[int] = None, **kw):
    """Functional shortcut: ``sa(f, bounds)`` -> :class:`Result`."""
    return SimulatedAnnealing(**kw).minimize(func, bounds, max_iter=max_iter, seed=seed)
