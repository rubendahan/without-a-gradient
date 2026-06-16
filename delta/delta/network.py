"""Road-network data model for the Delta traffic-signal problem.

This module defines the *static* description of a city: its signalised
intersections, the traffic movements (turning streams) that pass through each
one, and the demand on those movements. It contains no optimisation logic and
no delay maths, it is purely the structure that everything else is built on.

Traffic-engineering vocabulary used throughout the package
----------------------------------------------------------
* **Intersection**, a junction controlled by a traffic signal. Its timing is
  what we optimise.
* **Phase**, a set of movements that get a green light *at the same time*
  because they do not conflict (e.g. the two opposing through movements on the
  main street). A signal cycles through its phases one after another.
* **Cycle length** ``C`` (seconds), the time for one full pass through all
  phases. Typical urban values are 60-120 s.
* **Green split** ``g_i``, the fraction of the cycle that phase ``i`` is
  green. The splits of an intersection sum to 1 (we ignore inter-green/amber
  time for the proxy; it can be folded into a lost-time term if needed).
* **Offset** ``o`` (seconds, in ``[0, C)``), how far this intersection's
  cycle is shifted relative to a global clock. Coordinating offsets along a
  corridor creates a "green wave" so a platoon of cars meets successive greens.
* **Saturation flow** ``s`` (vehicles/second), the maximum discharge rate of
  a movement once it has a green and a queue, i.e. how fast cars leave when the
  light turns green. A typical lane is ~1800 veh/h = 0.5 veh/s.
* **Arrival rate / demand** ``q`` (vehicles/second), how fast cars arrive at
  the movement.

The ratio ``x = q / (s * g)`` is the movement's **degree of saturation**. When
``x < 1`` the movement can clear its demand within the green; as ``x -> 1`` the
queue stops clearing within one cycle and delay blows up. This single quantity
is what the whole Delta finding turns on (see :mod:`delta.analysis`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class Movement:
    """A single turning stream through an intersection (e.g. eastbound-through).

    Attributes
    ----------
    name:
        Human-readable label, handy for diagnostics.
    arrival_rate:
        Demand ``q`` in vehicles/second arriving at this movement.
    saturation_flow:
        Saturation flow ``s`` in vehicles/second, the discharge rate during
        green. Must be > 0.
    """

    name: str
    arrival_rate: float       # q  [veh/s]
    saturation_flow: float    # s  [veh/s]


@dataclass
class Phase:
    """A group of non-conflicting movements served together by one green.

    All movements in a phase receive green simultaneously. A signal serves its
    phases in order, once per cycle.
    """

    name: str
    movements: List[Movement] = field(default_factory=list)

    @property
    def total_arrival(self) -> float:
        """Sum of demand over the phase's movements [veh/s]."""
        return sum(m.arrival_rate for m in self.movements)

    @property
    def max_flow_ratio(self) -> float:
        """The critical ``q/s`` of the phase.

        A phase needs *at least* ``max(q/s)`` of the cycle as green just to keep
        up with its busiest movement. Summing this over phases gives the minimum
        feasible cycle utilisation; if it exceeds 1 the intersection is
        oversaturated no matter how the green is split.
        """
        return max((m.arrival_rate / m.saturation_flow for m in self.movements),
                   default=0.0)


@dataclass
class Intersection:
    """A signalised junction: an ordered list of phases plus timing bounds.

    The decision variables that the optimiser controls for this intersection are
    its per-phase green splits and its offset (and, if enabled globally, its
    cycle length). The bounds below are *physical* constraints that any valid
    plan must respect.

    Attributes
    ----------
    name:
        Identifier.
    phases:
        Ordered phases served once per cycle.
    cycle_length:
        Nominal cycle length ``C`` [s]. Used when cycle length is held fixed.
    min_green:
        Minimum green per phase [s] (pedestrian-clearance / safety floor). The
        per-phase split is never allowed below ``min_green / cycle_length``.
    cycle_bounds:
        ``(C_min, C_max)`` allowed cycle range [s] when cycle length is a
        decision variable.
    """

    name: str
    phases: List[Phase]
    cycle_length: float = 90.0
    min_green: float = 7.0
    cycle_bounds: tuple[float, float] = (60.0, 120.0)

    @property
    def n_phases(self) -> int:
        return len(self.phases)

    @property
    def critical_sum(self) -> float:
        """Sum of critical ``q/s`` ratios over all phases (a.k.a. flow ratio Y).

        Webster's theory says the intersection is undersaturated iff this sum is
        below 1 (strictly, below ``1 - lost_time/C``). It is the headline
        "how loaded is this junction" number.
        """
        return sum(p.max_flow_ratio for p in self.phases)


@dataclass
class RoadNetwork:
    """A whole city: a list of intersections plus corridor coordination links.

    Attributes
    ----------
    intersections:
        All signalised junctions, in a fixed order. That order defines the
        layout of the decision vector (see :mod:`delta.plan`).
    corridors:
        Each corridor is an ordered list of intersection indices that form a
        coordinated arterial. Adjacent pairs along a corridor are rewarded for
        good offset coordination (a green wave); see the offset-coordination
        term in :mod:`delta.simulator`.
    free_flow_travel_time:
        Nominal travel time [s] between adjacent intersections on a corridor,
        used to compute the *ideal* offset for a green wave. One scalar shared
        by all links keeps the example simple; a real network would store this
        per link.
    """

    intersections: List[Intersection]
    corridors: List[List[int]] = field(default_factory=list)
    free_flow_travel_time: float = 25.0

    @property
    def n_intersections(self) -> int:
        return len(self.intersections)

    @property
    def n_phases_total(self) -> int:
        return sum(i.n_phases for i in self.intersections)


def build_example_city(
    n_intersections: int = 40,
    seed: int = 0,
    load: float = 0.55,
) -> RoadNetwork:
    """Construct a small, reproducible example city as a stand-in for Delta's.

    The real Delta network had ~341 signals; this builds a smaller but
    structurally faithful grid-with-arterials so the package runs in seconds on
    a laptop. Pass a larger ``n_intersections`` (and see :mod:`delta.solver`)
    to scale up toward the real size.

    Parameters
    ----------
    n_intersections:
        Number of signalised junctions to generate.
    seed:
        RNG seed for reproducible demand.
    load:
        Target *mean* degree of saturation. With the default ``0.55`` the
        network is comfortably undersaturated, which, as the Delta team found,
        is exactly the regime where the optimisation ceiling is flat. Push it
        toward ``0.95`` to create a congested city where timing actually bites.

    Returns
    -------
    RoadNetwork
        A fully-populated network ready to be optimised.
    """
    rng = np.random.default_rng(seed)
    intersections: List[Intersection] = []

    # Each junction is a simple 2-phase signal: main street vs. cross street,
    # each phase carrying two opposing through movements. This is the canonical
    # textbook intersection and keeps the example readable while still exercising
    # every term in the delay model.
    # With a 2-phase signal the equal (all-green) split is g = 0.5 per phase.
    # We choose demand so the *main* street's mean degree of saturation under
    # that split equals ``load``: x = q / (s * g) = load  =>  q = load * s * g.
    equal_green = 0.5
    for k in range(n_intersections):
        sat = 0.5  # veh/s per movement (~1800 veh/h), a standard lane figure.
        phases = []
        for side in ("main", "cross"):
            # Cross streets carry less demand than the main arterial.
            base = load * sat * equal_green * (1.0 if side == "main" else 0.6)
            movements = [
                Movement(
                    name=f"int{k}-{side}-{d}",
                    # Jitter demand per movement so the city is not perfectly
                    # uniform; clip to stay non-negative.
                    arrival_rate=max(0.0, base * rng.uniform(0.7, 1.3)),
                    saturation_flow=sat,
                )
                for d in ("fwd", "rev")
            ]
            phases.append(Phase(name=f"int{k}-{side}", movements=movements))
        intersections.append(
            Intersection(name=f"int{k}", phases=phases, cycle_length=90.0)
        )

    # Lay the junctions out as a near-square grid and treat each grid row as a
    # coordinated arterial corridor (a "green wave" candidate).
    width = max(1, int(round(np.sqrt(n_intersections))))
    corridors: List[List[int]] = []
    for row_start in range(0, n_intersections, width):
        corridor = list(range(row_start, min(row_start + width, n_intersections)))
        if len(corridor) >= 2:
            corridors.append(corridor)

    return RoadNetwork(intersections=intersections, corridors=corridors)
