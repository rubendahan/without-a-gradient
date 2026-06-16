"""The flat-ceiling diagnostic, "characterize the objective before optimising".

This module reproduces the central finding of the Delta 2026 hackathon. After
throwing a whole toolbox of metaheuristics at the signal-timing problem, the
team's most valuable result was not a clever optimiser but a *characterisation
of the objective*:

    For the given demand, the network was so far below saturation that almost
    any sane plan was within <1% of optimal. The naive "all-green, never stop
    anyone" baseline was already essentially the answer.

The lesson, **characterise the objective before optimising it**, is what
this diagnostic demonstrates. :func:`characterize` compares the all-green
baseline against an optimised plan (and a random reference) and reports the gap.
On an undersaturated network that gap is tiny, exposing a flat optimisation
ceiling; push the network toward capacity (raise ``load`` in
:func:`delta.network.build_example_city`) and the gap opens up, showing that the
optimiser *does* earn its keep once timing actually matters.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .network import RoadNetwork, build_example_city
from .plan import SignalPlan
from .simulator import DelayProxy
from .solver import random_baseline, solve


@dataclass
class CharacterizationReport:
    """Numbers behind the flat-ceiling finding.

    The headline comparison is between the **sane plan** (Webster
    demand-proportional split, no optimisation at all) and the **optimised
    plan**. A tiny gap means a flat ceiling: the optimiser earns nothing.

    Attributes
    ----------
    all_green_delay:
        Total delay of the literal equal-split, zero-offset "all-green" plan,
        reported for context.
    sane_delay:
        Total delay of the demand-proportional "sane plan" (Webster's rule of
        thumb), the no-optimisation reference the finding is measured against.
    random_best_delay:
        Best total delay over a batch of random plans.
    optimized_delay:
        Total delay of the optimised plan.
    gap_fraction:
        Relative improvement of the optimised plan over the sane plan,
        ``(sane - optimized) / sane``. *This is the headline number*: small
        (<1%) means a flat ceiling; large means timing matters. It can go
        slightly negative when the optimiser, on a fixed budget, fails to even
        match the sane plan, the strongest possible sign of a flat ceiling.
    mean_saturation:
        Mean degree of saturation across movements, the regime indicator. Well
        below 1 explains a flat ceiling.
    """

    all_green_delay: float
    sane_delay: float
    random_best_delay: float
    optimized_delay: float
    gap_fraction: float
    mean_saturation: float

    def is_flat_ceiling(self, threshold: float = 0.01) -> bool:
        """True when optimisation buys less than ``threshold`` (default 1%)."""
        return self.gap_fraction < threshold

    def summary(self) -> str:
        """A human-readable report block, ready to print."""
        verdict = (
            "FLAT CEILING: optimisation buys < 1% over a sane plan, "
            "characterising the objective beat optimising it."
            if self.is_flat_ceiling()
            else "Timing matters: optimisation yields a meaningful improvement."
        )
        return (
            "Delta objective characterization\n"
            "--------------------------------\n"
            f"  mean degree of saturation : {self.mean_saturation:6.3f}  "
            f"({'undersaturated' if self.mean_saturation < 0.85 else 'near/over capacity'})\n"
            f"  all-green (equal) delay   : {self.all_green_delay:14.1f} veh-s\n"
            f"  sane plan delay (Webster) : {self.sane_delay:14.1f} veh-s\n"
            f"  random-search best delay  : {self.random_best_delay:14.1f} veh-s\n"
            f"  optimised plan delay      : {self.optimized_delay:14.1f} veh-s\n"
            f"  improvement over sane plan: {self.gap_fraction * 100:13.3f} %\n"
            f"  -> {verdict}"
        )


def mean_saturation(network: RoadNetwork) -> float:
    """Mean degree of saturation ``x = q / (s*g)`` under an equal-split plan.

    Computed at the all-green (equal-split) operating point, which is the
    natural reference. It is the single number that explains the ceiling: well
    below 1 means every movement clears its queue every cycle, so there is
    almost no delay to optimise away.
    """
    sats = []
    for inter in network.intersections:
        g = 1.0 / inter.n_phases  # equal split
        for phase in inter.phases:
            for mv in phase.movements:
                if mv.arrival_rate > 0:
                    sats.append(mv.arrival_rate / (mv.saturation_flow * g))
    return float(np.mean(sats)) if sats else 0.0


def characterize(
    network: Optional[RoadNetwork] = None,
    *,
    max_iter: int = 40,
    n_random: int = 100,
    seed: int = 0,
) -> CharacterizationReport:
    """Run the flat-ceiling diagnostic and return its numbers.

    Builds (or accepts) a network, evaluates the all-green baseline, a random
    search, and an optimised plan on the same :class:`DelayProxy`, and reports
    the gap. Swap the proxy for Delta's simulator (via :func:`delta.solver.solve`
    with ``simulator=``) to run the very same diagnostic on the real objective.
    """
    if network is None:
        network = build_example_city()

    plan = SignalPlan(network)
    sim = DelayProxy(network, plan)

    all_green_delay = sim.evaluate(plan.all_green())
    sane_delay = sim.evaluate(plan.proportional())
    random_best = random_baseline(network, sim, plan, n=n_random, seed=seed)

    outcome = solve(
        network, simulator=sim, max_iter=max_iter, seed=seed
    )
    optimized_delay = outcome.result.best_f

    # Improvement of the optimiser over the sane (no-optimisation) plan. Guard
    # against a zero baseline (an empty/degenerate network).
    denom = sane_delay if sane_delay > 0 else 1.0
    gap = (sane_delay - optimized_delay) / denom

    return CharacterizationReport(
        all_green_delay=all_green_delay,
        sane_delay=sane_delay,
        random_best_delay=random_best,
        optimized_delay=optimized_delay,
        gap_fraction=gap,
        mean_saturation=mean_saturation(network),
    )
