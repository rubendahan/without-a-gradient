"""Differential Evolution.

DE is a population method whose mutation operator is beautifully simple: it
perturbs one member using the *scaled difference of two others*. For the classic
``rand/1/bin`` strategy, each target ``x_i`` produces a donor

    v = x_a + F * (x_b - x_c)        (a, b, c distinct, random)

then a *binomial crossover* mixes the donor with the target into a trial vector
``u`` (each coordinate taken from ``v`` with probability ``CR``, with at least one
guaranteed). The trial replaces the target only if it is at least as good
(*greedy selection*).

The magic is *self-scaling*: early on the population is spread out, so the
difference ``x_b - x_c`` is large and DE takes big exploratory steps; as it
converges the differences shrink and the steps fine-tune automatically -- no
explicit step-size schedule needed. ``best/1/bin`` biases the donor toward the
incumbent for faster (greedier) convergence.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .core import Bounds, Optimizer


class DifferentialEvolution(Optimizer):
    """DE with ``rand/1/bin`` or ``best/1/bin`` and binomial crossover.

    Parameters
    ----------
    pop_size:
        Population size (>= 4 required for distinct donors).
    F:
        Differential weight scaling the difference vector, usually ``0.5-0.9``.
    CR:
        Crossover probability per coordinate, usually ``0.7-0.9``.
    strategy:
        ``"rand1"`` (robust, explorative) or ``"best1"`` (greedier).
    """

    name = "DE"

    def __init__(
        self,
        pop_size: int = 40,
        F: float = 0.7,
        CR: float = 0.9,
        strategy: str = "rand1",
    ):
        if strategy not in ("rand1", "best1"):
            raise ValueError("strategy must be 'rand1' or 'best1'")
        if pop_size < 4:
            raise ValueError("pop_size must be >= 4")
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.strategy = strategy

    def _run(self, func, bounds: Bounds, max_iter, rng, callback):
        n, d = self.pop_size, bounds.dim
        pop = bounds.sample(n, rng)
        fit = func.batch(pop)

        best_i = int(np.argmin(fit))
        best_x = pop[best_i].copy()
        best_f = float(fit[best_i])
        history = [best_f]

        idx_all = np.arange(n)
        for it in range(max_iter):
            for i in range(n):
                choices = idx_all[idx_all != i]
                a, b, c = pop[rng.choice(choices, size=3, replace=False)]
                base = best_x if self.strategy == "best1" else a
                donor = base + self.F * (b - c)
                donor = bounds.clip(donor)

                cross = rng.random(d) < self.CR
                cross[rng.integers(0, d)] = True  # ensure at least one gene changes
                trial = np.where(cross, donor, pop[i])

                ft = func(trial)
                if ft <= fit[i]:
                    pop[i] = trial
                    fit[i] = ft
                    if ft < best_f:
                        best_f, best_x = float(ft), trial.copy()

            history.append(best_f)
            self._emit(callback, best_x, best_f, history, it + 1, self.name)

        return best_x, best_f, history, False, {"strategy": self.strategy}


def de(func, bounds: Bounds, max_iter: int = 200, seed: Optional[int] = None, **kw):
    """Functional shortcut: ``de(f, bounds)`` -> :class:`Result`."""
    return DifferentialEvolution(**kw).minimize(func, bounds, max_iter=max_iter, seed=seed)
