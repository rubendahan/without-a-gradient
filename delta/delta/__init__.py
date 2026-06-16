"""Delta, traffic-signal retiming for the "Delta 2026" competition (Task D).

This package is our Delta 2026 competition team's deliverable (Delta was
sponsored by Mireo). It packages the
optimisation approach we used on Task D (retime a city's traffic signals to
minimise total vehicle delay over a 4-hour simulation) together with the
diagnostic that captures our key finding.

Layout
------
* :mod:`delta.network`   , the road-network data model and an example city.
* :mod:`delta.plan`      , decision-vector <-> structured signal-plan encoding.
* :mod:`delta.simulator` , ``DelayProxy``, a documented STAND-IN for Delta's
  proprietary simulator, exposing the same ``evaluate(plan) -> float`` query.
* :mod:`delta.solver`    , wires metaheuristics-lab (multi-population PSO) to
  the problem.
* :mod:`delta.analysis`  , the flat-ceiling diagnostic (our key finding).

IMPORTANT: :class:`delta.simulator.DelayProxy` is a transparent analytical
stand-in, **not** Delta's real simulator (which is proprietary and unavailable
to us). It mirrors the real query interface so the real engine can drop in
unchanged, see the SIMULATOR-SWAP note in :mod:`delta.simulator`.

Quick start
-----------
>>> from delta import build_example_city, solve, characterize
>>> report = characterize()
>>> print(report.summary())
"""
from .network import (
    Intersection,
    Movement,
    Phase,
    RoadNetwork,
    build_example_city,
)
from .plan import IntersectionTiming, SignalPlan
from .simulator import DelayProxy, Simulator
from .solver import SolveOutcome, random_baseline, solve
from .analysis import CharacterizationReport, characterize, mean_saturation

__version__ = "0.1.0"

__all__ = [
    # network
    "Movement", "Phase", "Intersection", "RoadNetwork", "build_example_city",
    # plan
    "SignalPlan", "IntersectionTiming",
    # simulator
    "DelayProxy", "Simulator",
    # solver
    "solve", "SolveOutcome", "random_baseline",
    # analysis
    "characterize", "CharacterizationReport", "mean_saturation",
    "__version__",
]
