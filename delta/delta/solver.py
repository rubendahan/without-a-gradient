"""Wire the metaheuristics-lab optimisers to the Delta signal-timing problem.

The optimisation engine lives in the separate ``metaheuristics`` package (the
team's metaheuristics-lab library), depended on via ``pyproject.toml``. This
module is just the glue: it builds the search box from a :class:`SignalPlan`,
hands the simulator's ``evaluate`` as the objective, and runs the optimiser.

The primary optimiser is the **multi-population PSO** (:class:`MultiSwarm`) that
the Delta team used: three sub-swarms with different explore/exploit balances,
exploratory, balanced, exploitative, sharing a global best. Other optimisers
from the library (GA, DE, SA, CMA-ES, Bayesian) implement the same
``minimize(func, bounds) -> Result`` API and can be dropped in via the
``optimizer=`` argument for comparison; that interchangeability is the whole
selling point of metaheuristics-lab.

Evaluation budget
-----------------
On the real problem each query costs ~5 s of wall-clock, so the number of
evaluations is the binding constraint. ``MultiSwarm`` makes
``n_swarms * n_particles * (max_iter + 1)`` calls; pick ``max_iter`` to fit your
budget. The proxy here is cheap, so the defaults run comfortably on a laptop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from metaheuristics import Bounds, MultiSwarm, Optimizer, Result

from .network import RoadNetwork, build_example_city
from .plan import SignalPlan
from .simulator import DelayProxy


@dataclass
class SolveOutcome:
    """Everything a caller needs after a solve, in one object.

    Attributes
    ----------
    result:
        The raw :class:`metaheuristics.Result` (best vector, value, history...).
    plan:
        The :class:`SignalPlan` encoder used (decode ``result.best_x`` with it).
    network:
        The optimised network.
    simulator:
        The objective that was evaluated.
    """

    result: Result
    plan: SignalPlan
    network: RoadNetwork
    simulator: object


def solve(
    network: Optional[RoadNetwork] = None,
    *,
    simulator: Optional[object] = None,
    optimizer: Optional[Optimizer] = None,
    optimize_cycle: bool = False,
    max_iter: int = 40,
    seed: int = 0,
) -> SolveOutcome:
    """Optimise the signal timings of ``network`` to minimise total delay.

    Parameters
    ----------
    network:
        City to optimise. Defaults to :func:`delta.network.build_example_city`.
    simulator:
        Any object with ``evaluate(plan_vector) -> float`` (lower = better).
        Defaults to the :class:`DelayProxy` stand-in. **Pass Delta's real
        simulator here to optimise the actual objective** (see the
        SIMULATOR-SWAP note in :mod:`delta.simulator`).
    optimizer:
        Any ``metaheuristics`` optimiser. Defaults to the multi-population PSO
        (:class:`metaheuristics.MultiSwarm`) the Delta team used.
    optimize_cycle:
        If True, cycle length becomes a decision variable too (otherwise it is
        held at each intersection's nominal value).
    max_iter:
        Optimiser iterations, the main knob on the evaluation budget.
    seed:
        RNG seed for a reproducible run.

    Returns
    -------
    SolveOutcome
        The result plus the encoder/network/simulator used.
    """
    if network is None:
        network = build_example_city()

    plan = SignalPlan(network, optimize_cycle=optimize_cycle)
    bounds: Bounds = plan.bounds()

    if simulator is None:
        simulator = DelayProxy(network, plan)
    if optimizer is None:
        # The Delta configuration: three sub-swarms sharing a global best.
        optimizer = MultiSwarm(n_swarms=3, n_particles=20, migrate_every=10)

    # The objective is just the simulator's scalar query. ``DelayProxy`` is
    # callable, but wrap explicitly so any plain object with ``.evaluate`` works.
    objective = getattr(simulator, "evaluate", simulator)

    result = optimizer.minimize(objective, bounds, max_iter=max_iter, seed=seed)
    return SolveOutcome(
        result=result, plan=plan, network=network, simulator=simulator
    )


def random_baseline(
    network: RoadNetwork,
    simulator: object,
    plan: SignalPlan,
    n: int = 50,
    seed: int = 0,
) -> float:
    """Best total delay found by ``n`` purely random plans, a sanity floor.

    A competent optimiser should at least match, and usually beat, the best of a
    handful of random plans. Used by the tests and the diagnostics as a
    reference point.
    """
    rng = np.random.default_rng(seed)
    bounds = plan.bounds()
    best = np.inf
    evaluate = getattr(simulator, "evaluate", simulator)
    for x in bounds.sample(n, rng):
        best = min(best, evaluate(x))
    return float(best)
