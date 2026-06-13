"""Hill climbing and random-restart hill climbing.

The simplest local search there is: from the current point, take a small random
step; keep it only if it improves. With a shrinking step it converges to the
nearest local optimum -- fast, but it has no way out of a basin once inside.

*Random-restart* hill climbing wraps that weakness in a loop: run many short
climbs from independent random starts and return the best summit found. It is the
honest baseline every fancier metaheuristic should be measured against -- if a
swarm or GA cannot beat random restarts, it is not earning its complexity.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .core import Bounds, Optimizer


class HillClimbing(Optimizer):
    """Stochastic hill climbing with an adaptive (shrinking) step size.

    Parameters
    ----------
    step_frac:
        Initial step std as a fraction of each axis span.
    shrink:
        Multiply the step by this after a failed move...
    grow:
        ...and by this after a successful one (so the step adapts to the local
        landscape -- the classic 1/5th-rule idea, simplified).
    restarts:
        Number of independent climbs; the best summit wins. ``1`` is plain HC.
    """

    name = "HillClimbing"

    def __init__(
        self,
        step_frac: float = 0.1,
        shrink: float = 0.9,
        grow: float = 1.1,
        restarts: int = 1,
    ):
        self.step_frac = step_frac
        self.shrink = shrink
        self.grow = grow
        self.restarts = restarts

    def _climb(self, func, bounds, iters, rng):
        x = bounds.sample(1, rng)[0]
        fx = func(x)
        step = self.step_frac * bounds.span
        local_hist = [fx]
        for _ in range(iters):
            cand = bounds.reflect(x + rng.normal(0.0, step))
            fc = func(cand)
            if fc < fx:
                x, fx = cand, fc
                step = step * self.grow
            else:
                step = step * self.shrink
            local_hist.append(fx)
        return x, fx, local_hist

    def _run(self, func, bounds: Bounds, max_iter, rng, callback):
        iters_each = max(1, max_iter // self.restarts)
        best_x, best_f = None, np.inf
        history = []
        for r in range(self.restarts):
            x, fx, local_hist = self._climb(func, bounds, iters_each, rng)
            if fx < best_f:
                best_f, best_x = fx, x.copy()
            # Stitch each climb into a single best-so-far curve.
            for h in local_hist:
                history.append(min(best_f, h))
            self._emit(callback, best_x, best_f, history, len(history), self.name)
        return best_x, best_f, history, False, {"restarts": self.restarts}


def hill_climbing(func, bounds: Bounds, max_iter: int = 500, seed: Optional[int] = None, **kw):
    """Functional shortcut: ``hill_climbing(f, bounds)`` -> :class:`Result`."""
    return HillClimbing(**kw).minimize(func, bounds, max_iter=max_iter, seed=seed)
