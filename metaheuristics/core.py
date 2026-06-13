"""Core abstractions shared by every optimizer in the library.

The whole package is built around three small objects:

* :class:`Bounds`  -- the box-constrained search space ``[lower, upper]^d``.
* :class:`Result`  -- a uniform return value (best point, best value, history...).
* :class:`Optimizer` -- the abstract base class every algorithm subclasses.

Every optimizer minimizes a scalar objective ``f: R^d -> R`` over a box.
Maximization is obtained for free by minimizing ``-f`` (see :func:`as_minimizer`).

The design goal is that swapping one algorithm for another is a one-line change::

    from metaheuristics import ParticleSwarm, GeneticAlgorithm, Bounds
    from metaheuristics.benchmarks import rastrigin

    bounds = Bounds([-5.12] * 2, [5.12] * 2)
    res_a = ParticleSwarm().minimize(rastrigin, bounds, seed=0)
    res_b = GeneticAlgorithm().minimize(rastrigin, bounds, seed=0)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

Objective = Callable[[np.ndarray], float]
Callback = Callable[["Result"], None]


class Bounds:
    """A box-constrained search space ``[lower_i, upper_i]`` for ``i in 0..d-1``.

    Parameters
    ----------
    lower, upper:
        Per-dimension lower/upper limits. Scalars are broadcast when ``dim`` is
        given, e.g. ``Bounds(-5, 5, dim=10)``.
    """

    def __init__(
        self,
        lower: Sequence[float] | float,
        upper: Sequence[float] | float,
        dim: Optional[int] = None,
    ):
        lower = np.atleast_1d(np.asarray(lower, dtype=float))
        upper = np.atleast_1d(np.asarray(upper, dtype=float))
        if dim is not None:
            if lower.size == 1:
                lower = np.full(dim, lower.item())
            if upper.size == 1:
                upper = np.full(dim, upper.item())
        if lower.shape != upper.shape:
            raise ValueError(f"lower {lower.shape} and upper {upper.shape} differ")
        if np.any(upper < lower):
            raise ValueError("upper bound must be >= lower bound on every axis")
        self.lower = lower
        self.upper = upper

    @property
    def dim(self) -> int:
        return self.lower.size

    @property
    def span(self) -> np.ndarray:
        return self.upper - self.lower

    def clip(self, x: np.ndarray) -> np.ndarray:
        """Project points back into the box (works on a single point or a batch)."""
        return np.clip(x, self.lower, self.upper)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` points uniformly at random inside the box -> ``(n, d)``."""
        return self.lower + rng.random((n, self.dim)) * self.span

    def reflect(self, x: np.ndarray) -> np.ndarray:
        """Reflect out-of-box coordinates back inside (gentler than hard clipping)."""
        lo, hi, span = self.lower, self.upper, self.span
        y = x - lo
        two = 2.0 * span
        y = np.mod(y, two)
        y = np.where(y > span, two - y, y)
        return lo + y

    def __repr__(self) -> str:
        return f"Bounds(dim={self.dim}, lower={self.lower}, upper={self.upper})"


@dataclass
class Result:
    """Uniform result object returned by every :meth:`Optimizer.minimize` call."""

    best_x: np.ndarray
    best_f: float
    history: list = field(default_factory=list)  # best-so-far value per iteration
    n_evals: int = 0
    n_iter: int = 0
    converged: bool = False
    name: str = ""
    meta: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        x = np.array2string(np.asarray(self.best_x), precision=4, suppress_small=True)
        return (
            f"Result(name={self.name!r}, best_f={self.best_f:.6g}, "
            f"best_x={x}, n_evals={self.n_evals}, n_iter={self.n_iter}, "
            f"converged={self.converged})"
        )


class _EvalCounter:
    """Wraps an objective to count evaluations and track the running best."""

    def __init__(self, func: Objective):
        self._func = func
        self.n_evals = 0
        self.best_f = np.inf
        self.best_x: Optional[np.ndarray] = None

    def __call__(self, x: np.ndarray) -> float:
        f = float(self._func(x))
        self.n_evals += 1
        if f < self.best_f:
            self.best_f = f
            self.best_x = np.array(x, dtype=float, copy=True)
        return f

    def batch(self, X: np.ndarray) -> np.ndarray:
        """Evaluate a ``(n, d)`` batch -> ``(n,)`` values, updating counters."""
        out = np.empty(len(X))
        for i, x in enumerate(X):
            out[i] = self(x)
        return out


class Optimizer:
    """Abstract base class for all optimizers.

    Subclasses implement :meth:`_run`. They get a wrapped, eval-counting objective
    plus a seeded NumPy ``Generator`` and should return ``(best_x, best_f, history,
    converged, meta)``. The :meth:`minimize` wrapper handles bookkeeping and the
    uniform :class:`Result`.
    """

    name = "optimizer"

    def minimize(
        self,
        func: Objective,
        bounds: Bounds,
        max_iter: int = 200,
        seed: Optional[int] = None,
        callback: Optional[Callback] = None,
    ) -> Result:
        rng = np.random.default_rng(seed)
        counter = _EvalCounter(func)
        best_x, best_f, history, converged, meta = self._run(
            counter, bounds, max_iter, rng, callback
        )
        return Result(
            best_x=np.asarray(best_x, dtype=float),
            best_f=float(best_f),
            history=list(history),
            n_evals=counter.n_evals,
            n_iter=len(history),
            converged=bool(converged),
            name=self.name,
            meta=meta or {},
        )

    # Subclasses must implement this.
    def _run(self, func, bounds, max_iter, rng, callback):  # pragma: no cover
        raise NotImplementedError

    @staticmethod
    def _emit(callback, best_x, best_f, history, n_iter, name):
        if callback is None:
            return
        callback(
            Result(
                best_x=np.asarray(best_x, dtype=float),
                best_f=float(best_f),
                history=list(history),
                n_iter=n_iter,
                name=name,
            )
        )


def as_minimizer(func: Objective, maximize: bool = False) -> Objective:
    """Return ``func`` for minimization, or ``-func`` when ``maximize`` is True."""
    if not maximize:
        return func
    return lambda x: -func(x)
