"""Standard global-optimization test functions.

Each function ``f: R^d -> R`` takes a 1-D NumPy array and returns a scalar.
All are written to work in any dimension unless noted, and each carries metadata
(global minimum, recommended search box) accessible through :data:`BENCHMARKS`.

These are the classic landscapes used to stress-test metaheuristics:

* **sphere**     -- convex bowl, the easy sanity check.
* **rosenbrock** -- the banana valley; gradient methods love it, samplers struggle.
* **rastrigin**  -- a sphere wrapped in a cosine egg-carton: ~10^d local minima.
* **ackley**     -- nearly flat outer plateau with a deep central funnel.
* **griewank**   -- product of cosines; deceptively multimodal at small scale.
* **schwefel**   -- global optimum sits far from the centre, near the box corner.

Reference: Jamil & Yang, "A literature survey of benchmark functions for global
optimization problems" (2013).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Benchmark:
    """A test function bundled with the facts you need to evaluate a solver."""

    func: callable
    lower: float
    upper: float
    f_min: float          # known global minimum value
    x_min: str            # human description of the optimizer's location
    multimodal: bool
    description: str

    def __call__(self, x):
        return self.func(np.asarray(x, dtype=float))


def sphere(x: np.ndarray) -> float:
    return float(np.sum(x * x))


def rosenbrock(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


def rastrigin(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(10.0 * x.size + np.sum(x * x - 10.0 * np.cos(2.0 * np.pi * x)))


def ackley(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    d = x.size
    s1 = np.sum(x * x)
    s2 = np.sum(np.cos(2.0 * np.pi * x))
    return float(
        -20.0 * np.exp(-0.2 * np.sqrt(s1 / d))
        - np.exp(s2 / d)
        + 20.0
        + np.e
    )


def griewank(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    i = np.arange(1, x.size + 1)
    return float(1.0 + np.sum(x * x) / 4000.0 - np.prod(np.cos(x / np.sqrt(i))))


def schwefel(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(418.9828872724338 * x.size - np.sum(x * np.sin(np.sqrt(np.abs(x)))))


BENCHMARKS: dict[str, Benchmark] = {
    "sphere": Benchmark(
        sphere, -5.12, 5.12, 0.0, "origin (0,...,0)", False,
        "Convex bowl. The sanity check every solver must pass.",
    ),
    "rosenbrock": Benchmark(
        rosenbrock, -2.048, 2.048, 0.0, "(1,...,1)", False,
        "The banana valley: a narrow, curved, near-flat ridge.",
    ),
    "rastrigin": Benchmark(
        rastrigin, -5.12, 5.12, 0.0, "origin (0,...,0)", True,
        "Sphere + cosine ripples: a regular grid of ~10^d local minima.",
    ),
    "ackley": Benchmark(
        ackley, -32.768, 32.768, 0.0, "origin (0,...,0)", True,
        "A wide near-flat plateau hiding one deep central funnel.",
    ),
    "griewank": Benchmark(
        griewank, -600.0, 600.0, 0.0, "origin (0,...,0)", True,
        "Product of cosines; many local minima that flatten out at scale.",
    ),
    "schwefel": Benchmark(
        schwefel, -500.0, 500.0, 0.0, "(420.9687,...) near a corner", True,
        "Deceptive: the global optimum sits far from the centre.",
    ),
}


def get(name: str) -> Benchmark:
    """Look up a benchmark by name (case-insensitive)."""
    try:
        return BENCHMARKS[name.lower()]
    except KeyError as exc:
        raise KeyError(
            f"unknown benchmark {name!r}; choose from {sorted(BENCHMARKS)}"
        ) from exc


__all__ = [
    "Benchmark", "BENCHMARKS", "get",
    "sphere", "rosenbrock", "rastrigin", "ackley", "griewank", "schwefel",
]
