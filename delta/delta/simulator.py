"""DelayProxy, a self-contained mesoscopic delay model (a STAND-IN simulator).

=============================================================================
  THIS IS A PROXY, NOT DELTA'S SIMULATOR.
=============================================================================
The real Delta 2026 objective was the competition's proprietary *mesoscopic*
traffic simulator: you submit a full signal plan and it simulates ~5,215 vehicles
over a 4-hour horizon and returns ONE scalar, the total vehicle delay. That
simulator is not available to us, so this module provides a transparent,
NumPy-only **analytical delay model** with the *same query interface*:

    evaluate(plan_vector: np.ndarray) -> float        # total delay [veh-seconds]

Everything downstream (the encoding, the solver, the diagnostics) talks to that
one method. To plug in the real engine, see :class:`Simulator` and the
``SIMULATOR-SWAP`` note at the bottom of this file, you only have
to implement the same ``evaluate`` signature.

The delay model
---------------
We use a classic **Webster / HCM-style intersection delay** decomposed into two
physically meaningful parts, evaluated per movement and summed:

1. **Uniform delay**, the unavoidable wait from arriving on red even when
   demand is perfectly steady. For an arrival rate ``q``, green split ``g`` and
   cycle ``C`` it is the well-known

       d_uniform = 0.5 * C * (1 - g)^2 / (1 - min(x, 1) * g)

   where ``x = q / (s * g)`` is the degree of saturation. More green (larger
   ``g``) shortens the red the movement waits on, lowering this term.

2. **Overflow / random delay**, extra delay once the movement approaches
   capacity and queues fail to clear each cycle. We use the standard
   incremental term

       d_overflow = 900 * T * [ (x - 1) + sqrt((x - 1)^2 + (8*k*x)/(c*T)) ]

   (HCM-style, with ``c = s*g`` the capacity and ``T`` the analysis period in
   hours). This is ~0 while ``x < 1`` and explodes as ``x -> 1`` and beyond,
   the hallmark of saturation.

On top of the per-intersection delay we add a small **coordination term**: along
each corridor, adjacent intersections whose offsets form a good "green wave"
(offset difference close to the free-flow travel time) get a delay *rebate*,
and badly-coordinated ones a small *penalty*. This is what makes offsets matter
and rewards arterial coordination, mirroring why the real simulator cares about
offsets at all.

Why a proxy is honest and useful here
-------------------------------------
The point of the Delta deliverable is the *optimisation methodology* and the
*finding* about the objective's shape, not a re-implementation of Delta's
physics. This proxy reproduces the qualitative behaviour that drove the finding
-- delay is flat and low while the network is undersaturated, and rises sharply
only near capacity, so the flat-ceiling diagnostic in :mod:`delta.analysis` is
demonstrated on a model we can fully explain. Swap in the real simulator and the
same code runs unchanged.
"""
from __future__ import annotations

import numpy as np

from .network import RoadNetwork
from .plan import SignalPlan

# Analysis period for the overflow term, in hours. The Delta horizon was 4 h.
_ANALYSIS_PERIOD_H = 4.0
# Incremental-delay calibration constant (HCM: ~0.5 for fixed-time signals).
_OVERFLOW_K = 0.5


class DelayProxy:
    """Analytical stand-in for Delta's mesoscopic simulator.

    Construct it from a :class:`RoadNetwork` and a :class:`SignalPlan` encoder,
    then call :meth:`evaluate` on a decision vector to get total delay. The class
    deliberately exposes exactly the surface the real simulator would, so it can
    be swapped out with no change to the optimiser or diagnostics.

    Parameters
    ----------
    network:
        The city to simulate.
    plan:
        The encoder mapping decision vectors to per-intersection timings. If
        omitted, a default :class:`SignalPlan` over ``network`` is created.
    coordination_weight:
        Strength of the green-wave reward/penalty relative to intersection
        delay. Set to 0 to disable corridor coordination entirely.
    """

    def __init__(
        self,
        network: RoadNetwork,
        plan: SignalPlan | None = None,
        coordination_weight: float = 1.0,
    ):
        self.network = network
        self.plan = plan if plan is not None else SignalPlan(network)
        self.coordination_weight = coordination_weight
        self.n_evals = 0  # query counter, evaluations are "expensive" by design

    #, the public, simulator-shaped query interface -------------------------

    def evaluate(self, plan_vector: np.ndarray) -> float:
        """Total vehicle delay [veh-seconds] for a decision vector. Lower = better.

        This is THE objective handed to the optimiser. It mirrors the real
        simulator's contract: one decision vector in, one scalar out.
        """
        self.n_evals += 1
        timings = self.plan.from_vector(plan_vector)

        total = 0.0
        for inter, t in zip(self.network.intersections, timings):
            for phase, g in zip(inter.phases, t.splits):
                for mv in phase.movements:
                    total += self._movement_delay(
                        q=mv.arrival_rate,
                        s=mv.saturation_flow,
                        g=g,
                        C=t.cycle_length,
                    )

        total += self.coordination_weight * self._coordination_delay(timings)
        return float(total)

    # Make the proxy directly callable, so it can be passed anywhere a plain
    # ``func(x) -> float`` objective is expected (e.g. metaheuristics.minimize).
    __call__ = evaluate

    #, the delay physics ----------------------------------------------------

    @staticmethod
    def _movement_delay(q: float, s: float, g: float, C: float) -> float:
        """Total delay rate for one movement [veh-seconds], Webster + overflow.

        Parameters mirror the module docstring: ``q`` arrivals/s, ``s``
        saturation flow veh/s, ``g`` green fraction, ``C`` cycle seconds.
        Returns the delay summed over all vehicles arriving in the horizon
        (per-vehicle delay times the number of arrivals), so plans are compared
        on the same total-delay scale the real simulator reports.
        """
        if q <= 0.0:
            return 0.0
        g = float(np.clip(g, 1e-6, 1.0))
        capacity = s * g                      # c = s*g  [veh/s]
        x = q / capacity                      # degree of saturation

        # 1) Uniform delay (per vehicle). The denominator uses min(x,1) so it
        #    stays finite at and beyond saturation; the overflow term then takes
        #    over to represent the unbounded queue growth.
        x_eff = min(x, 1.0)
        denom = max(1.0 - x_eff * g, 1e-6)
        d_uniform = 0.5 * C * (1.0 - g) ** 2 / denom

        # 2) Overflow / random delay (per vehicle), HCM incremental form.
        T = _ANALYSIS_PERIOD_H
        c_per_h = capacity * 3600.0           # capacity in veh/h for the formula
        root = (x - 1.0) ** 2 + (8.0 * _OVERFLOW_K * x) / (c_per_h * T)
        d_overflow = 900.0 * T * ((x - 1.0) + np.sqrt(root))

        per_vehicle = max(d_uniform + d_overflow, 0.0)

        # Number of vehicles arriving over the horizon = q * horizon_seconds.
        n_vehicles = q * (T * 3600.0)
        return per_vehicle * n_vehicles

    def _coordination_delay(self, timings) -> float:
        """Green-wave reward/penalty summed over corridor links [veh-seconds].

        For each adjacent pair on a corridor we compare the *actual* offset
        difference to the *ideal* one (the free-flow travel time between them,
        reduced modulo the cycle). A near-ideal offset lets a platoon roll
        through on green and earns a delay rebate; a worst-case offset (half a
        cycle off) incurs a penalty. The magnitude scales with the platoon size
        carried on the link.
        """
        net = self.network
        if not net.corridors:
            return 0.0

        ideal = net.free_flow_travel_time
        total = 0.0
        for corridor in net.corridors:
            for a, b in zip(corridor[:-1], corridor[1:]):
                ta, tb = timings[a], timings[b]
                C = 0.5 * (ta.cycle_length + tb.cycle_length)
                # Offset error wrapped to [-C/2, C/2]: how far from the ideal
                # green-wave offset this pair is.
                err = ((tb.offset - ta.offset) - ideal) % C
                if err > C / 2:
                    err -= C
                # Map |err| in [0, C/2] to a factor in [-1, +1]: perfectly
                # coordinated -> -1 (rebate), worst case -> +1 (penalty).
                factor = (abs(err) / (C / 2)) * 2.0 - 1.0
                # Platoon scale: demand on the upstream main phase over an hour.
                upstream = self.network.intersections[a]
                platoon = upstream.phases[0].total_arrival * 3600.0
                total += factor * platoon * 0.5  # 0.5 s of swing per vehicle
        return total


class Simulator:
    """Abstract query interface a city simulator must satisfy.

    Both :class:`DelayProxy` (this package's stand-in) and Delta's real engine
    are *Simulators* in the sense below. The optimiser and diagnostics depend
    only on this one method, so either can be dropped in interchangeably.

    .. rubric:: SIMULATOR-SWAP, how to plug in the real simulator

    Implement a class with the single method::

        def evaluate(self, plan_vector: numpy.ndarray) -> float:
            '''Run the city for the full horizon and return total vehicle
            delay in vehicle-seconds (lower is better).'''

    ``plan_vector`` is the unit-cube decision vector produced by
    :class:`delta.plan.SignalPlan`; decode it with ``plan.from_vector(...)`` to
    recover per-intersection green splits, offsets and cycle lengths exactly as
    this proxy does. Then pass an instance to :func:`delta.solver.solve` via its
    ``simulator=`` argument. Nothing else changes.
    """

    def evaluate(self, plan_vector: np.ndarray) -> float:  # pragma: no cover
        raise NotImplementedError(
            "Provide Delta's mesoscopic simulator here; see the SIMULATOR-SWAP "
            "note in delta/simulator.py."
        )
