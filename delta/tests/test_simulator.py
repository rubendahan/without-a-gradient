"""Tests for the delay proxy (delta.simulator).

These check the *qualitative physics* the proxy must obey, the same sanity
properties Delta's real simulator would satisfy: non-negative delay, more
saturation -> more delay, and starving a movement of green is terrible.
"""
from __future__ import annotations

import numpy as np

from delta.network import build_example_city
from delta.plan import SignalPlan
from delta.simulator import DelayProxy


def test_delay_is_non_negative():
    """Total delay is never negative for valid plans."""
    city = build_example_city(n_intersections=20, seed=0)
    plan = SignalPlan(city)
    sim = DelayProxy(city, plan)
    rng = np.random.default_rng(0)
    for x in plan.bounds().sample(10, rng):
        assert sim.evaluate(x) >= 0.0


def test_more_saturation_means_more_delay():
    """A more heavily loaded network has higher all-green delay (monotonicity)."""
    plan_args = dict(n_intersections=20, seed=0)
    delays = []
    for load in (0.4, 0.6, 0.8):
        city = build_example_city(load=load, **plan_args)
        plan = SignalPlan(city)
        sim = DelayProxy(city, plan)
        delays.append(sim.evaluate(plan.all_green()))
    assert delays[0] < delays[1] < delays[2]


def test_movement_delay_rises_toward_capacity():
    """Per-movement delay increases as the degree of saturation approaches 1."""
    # Fixed s, C, g; sweep arrival rate q upward -> x = q/(s*g) rises.
    s, C, g = 0.5, 90.0, 0.5
    prev = -1.0
    for q in (0.05, 0.10, 0.15, 0.20, 0.24):  # x up to ~0.96
        d = DelayProxy._movement_delay(q=q, s=s, g=g, C=C)
        assert d > prev
        prev = d


def test_all_red_is_terrible():
    """Starving phases (near-zero green) costs far more than a balanced plan.

    We cannot set an exact 'all red' through the encoder (min-green floors it),
    so we compare a near-degenerate split against the balanced all-green plan and
    require it to be much worse, the qualitative 'don't stop everyone' check.
    """
    city = build_example_city(n_intersections=10, seed=0, load=0.8)
    plan = SignalPlan(city)
    sim = DelayProxy(city, plan)

    balanced = sim.evaluate(plan.all_green())

    # Build a pathological vector: drive the first phase's weight to ~0 at every
    # intersection so the cross phase hogs the green and the main phase starves.
    bad = plan.all_green().copy()
    for k, inter in enumerate(city.intersections):
        start = int(plan._offsets[k])
        bad[start] = 0.0  # main-phase weight -> 0 (floored to min-green inside)
    starved = sim.evaluate(bad)

    assert starved > balanced


def test_callable_matches_evaluate():
    """Calling the proxy directly is identical to evaluate()."""
    city = build_example_city(n_intersections=8, seed=0)
    plan = SignalPlan(city)
    sim = DelayProxy(city, plan)
    x = plan.all_green()
    assert sim(x) == sim.evaluate(x)


def test_coordination_changes_objective():
    """Offsets matter: turning coordination on/off changes the delay."""
    city = build_example_city(n_intersections=16, seed=0)
    plan = SignalPlan(city)
    rng = np.random.default_rng(1)
    x = plan.bounds().sample(1, rng)[0]
    with_coord = DelayProxy(city, plan, coordination_weight=1.0).evaluate(x)
    without = DelayProxy(city, plan, coordination_weight=0.0).evaluate(x)
    assert with_coord != without
