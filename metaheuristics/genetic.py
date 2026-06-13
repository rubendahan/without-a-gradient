"""Genetic Algorithm for continuous (real-coded) optimization.

A GA evolves a *population* of candidate solutions through three operators that
imitate natural selection:

1. **Selection** -- pick parents biased toward fitness. Tournament selection
   draws ``k`` random individuals and keeps the best; ``k`` tunes the pressure.
2. **Crossover** -- recombine two parents into offspring. We offer *uniform*
   (swap genes coordinate-wise) and *BLX-alpha* (blend that can interpolate and
   slightly extrapolate between parents -- the standard real-coded operator).
3. **Mutation** -- perturb genes at random to inject diversity. We use Gaussian
   mutation with a per-coordinate sigma that anneals as the run progresses.

**Elitism** copies the best ``elite`` individuals unchanged into the next
generation so the incumbent can never be lost.

The interplay is the whole point: crossover *exploits* by mixing good building
blocks, mutation *explores* by stepping off the current population, and selection
applies the pressure that turns drift into progress.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .core import Bounds, Optimizer


class GeneticAlgorithm(Optimizer):
    """Real-coded GA with tournament selection, crossover, and elitism.

    Parameters
    ----------
    pop_size:
        Number of individuals per generation.
    tournament_k:
        Tournament size. Larger -> stronger selection pressure, faster but
        more prone to premature convergence.
    crossover:
        ``"uniform"`` or ``"blx"`` (BLX-alpha blend crossover).
    crossover_rate:
        Probability that a mating produces recombined offspring (else parents
        are copied through).
    blx_alpha:
        Blend factor for BLX-alpha; ``0.5`` allows mild extrapolation.
    mutation_rate:
        Per-gene probability of Gaussian mutation.
    sigma_frac:
        Initial mutation std as a fraction of each axis span.
    sigma_decay:
        Multiplicative factor applied to sigma each generation (annealing).
    elite:
        Number of top individuals copied unchanged each generation.
    """

    name = "GA"

    def __init__(
        self,
        pop_size: int = 60,
        tournament_k: int = 3,
        crossover: str = "blx",
        crossover_rate: float = 0.9,
        blx_alpha: float = 0.5,
        mutation_rate: float = 0.1,
        sigma_frac: float = 0.1,
        sigma_decay: float = 0.99,
        elite: int = 2,
    ):
        if crossover not in ("uniform", "blx"):
            raise ValueError("crossover must be 'uniform' or 'blx'")
        self.pop_size = pop_size
        self.tournament_k = tournament_k
        self.crossover = crossover
        self.crossover_rate = crossover_rate
        self.blx_alpha = blx_alpha
        self.mutation_rate = mutation_rate
        self.sigma_frac = sigma_frac
        self.sigma_decay = sigma_decay
        self.elite = elite

    def _tournament(self, fit, rng):
        contestants = rng.integers(0, len(fit), size=self.tournament_k)
        return int(contestants[np.argmin(fit[contestants])])

    def _cross(self, a, b, rng):
        if rng.random() >= self.crossover_rate:
            return a.copy(), b.copy()
        if self.crossover == "uniform":
            mask = rng.random(a.size) < 0.5
            c1 = np.where(mask, a, b)
            c2 = np.where(mask, b, a)
            return c1, c2
        # BLX-alpha: sample children from an interval extended by alpha on each side.
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        span = hi - lo
        ext_lo = lo - self.blx_alpha * span
        ext_hi = hi + self.blx_alpha * span
        c1 = ext_lo + rng.random(a.size) * (ext_hi - ext_lo)
        c2 = ext_lo + rng.random(a.size) * (ext_hi - ext_lo)
        return c1, c2

    def _run(self, func, bounds: Bounds, max_iter, rng, callback):
        d = bounds.dim
        pop = bounds.sample(self.pop_size, rng)
        fit = func.batch(pop)
        sigma = self.sigma_frac * bounds.span

        best_i = int(np.argmin(fit))
        best_x = pop[best_i].copy()
        best_f = float(fit[best_i])
        history = [best_f]

        for it in range(max_iter):
            order = np.argsort(fit)
            new_pop = [pop[order[e]].copy() for e in range(self.elite)]

            while len(new_pop) < self.pop_size:
                pa = pop[self._tournament(fit, rng)]
                pb = pop[self._tournament(fit, rng)]
                c1, c2 = self._cross(pa, pb, rng)
                for child in (c1, c2):
                    mask = rng.random(d) < self.mutation_rate
                    child = child + mask * rng.normal(0.0, sigma)
                    new_pop.append(bounds.clip(child))
                    if len(new_pop) >= self.pop_size:
                        break

            pop = np.array(new_pop[: self.pop_size])
            fit = func.batch(pop)
            sigma = sigma * self.sigma_decay

            gi = int(np.argmin(fit))
            if fit[gi] < best_f:
                best_f = float(fit[gi])
                best_x = pop[gi].copy()

            history.append(best_f)
            self._emit(callback, best_x, best_f, history, it + 1, self.name)

        return best_x, best_f, history, False, {"pop_size": self.pop_size}


def ga(func, bounds: Bounds, max_iter: int = 200, seed: Optional[int] = None, **kw):
    """Functional shortcut: ``ga(f, bounds)`` -> :class:`Result`."""
    return GeneticAlgorithm(**kw).minimize(func, bounds, max_iter=max_iter, seed=seed)
