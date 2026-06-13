"""Particle Swarm Optimization.

PSO models a population of candidate solutions as *particles* flying through the
search space. Each particle ``i`` remembers its own best-ever position
``p_i`` (the *cognitive* memory) and is pulled toward the swarm's best-known
position ``g`` (the *social* memory). At every step its velocity is updated by

    v <- w*v  +  c1*r1*(p_i - x_i)  +  c2*r2*(g - x_i)
    x <- x + v

with ``r1, r2 ~ U(0,1)`` drawn independently per coordinate. ``w`` is the
*inertia* (momentum), ``c1`` the cognitive weight, ``c2`` the social weight.

This module implements the canonical algorithm plus the refinements that made it
work on a real 341-dimensional problem:

* a **linear inertia schedule** ``w: w_start -> w_end`` (explore early, exploit late);
* optional **ring topology** -- a particle follows the best of its ``k`` ring
  neighbours instead of the global best, which slows premature convergence;
* **velocity clamping** to a fraction of the box span;
* **turbulent re-init** of the worst particles after a stagnation window;
* a :class:`MultiSwarm` that runs several swarms with different explore/exploit
  balances and shares their global best -- the configuration used on Delta.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .core import Bounds, Optimizer


class ParticleSwarm(Optimizer):
    """Canonical PSO with an inertia schedule and optional ring topology.

    Parameters
    ----------
    n_particles:
        Swarm size. 20-40 is typical; scale up with dimension.
    w_start, w_end:
        Inertia at the first and last iteration (linearly interpolated). High
        early inertia explores; low late inertia exploits.
    c1, c2:
        Cognitive (personal-best) and social (global-best) acceleration weights.
        The classic stable setting is ``c1 = c2 = 2.05`` (or ``2.0``).
    topology:
        ``"global"`` (gbest) or ``"ring"`` (lbest with ``ring_k`` neighbours).
    v_max_frac:
        Velocity clamp as a fraction of each axis span.
    stagnation:
        Re-randomize the worst ``turbulence_frac`` particles after this many
        iterations without global improvement. ``0`` disables it.
    tol:
        Stop early once the best value improves by less than ``tol`` over a
        window of iterations.
    """

    name = "PSO"

    def __init__(
        self,
        n_particles: int = 30,
        w_start: float = 0.9,
        w_end: float = 0.4,
        c1: float = 2.05,
        c2: float = 2.05,
        topology: str = "global",
        ring_k: int = 2,
        v_max_frac: float = 0.2,
        stagnation: int = 20,
        turbulence_frac: float = 0.2,
        tol: float = 1e-10,
    ):
        if topology not in ("global", "ring"):
            raise ValueError("topology must be 'global' or 'ring'")
        self.n_particles = n_particles
        self.w_start = w_start
        self.w_end = w_end
        self.c1 = c1
        self.c2 = c2
        self.topology = topology
        self.ring_k = ring_k
        self.v_max_frac = v_max_frac
        self.stagnation = stagnation
        self.turbulence_frac = turbulence_frac
        self.tol = tol

    def _ring_best(self, pbest_x, pbest_f):
        """For each particle, the best personal-best among its ring neighbours."""
        n = len(pbest_f)
        out = np.empty_like(pbest_x)
        k = self.ring_k
        for i in range(n):
            idx = [(i + off) % n for off in range(-k, k + 1)]
            j = idx[int(np.argmin(pbest_f[idx]))]
            out[i] = pbest_x[j]
        return out

    def _run(self, func, bounds: Bounds, max_iter, rng, callback):
        d = bounds.dim
        n = self.n_particles
        v_max = self.v_max_frac * bounds.span

        x = bounds.sample(n, rng)
        v = (rng.random((n, d)) * 2 - 1) * v_max
        f = func.batch(x)

        pbest_x = x.copy()
        pbest_f = f.copy()
        g_i = int(np.argmin(pbest_f))
        gbest_x = pbest_x[g_i].copy()
        gbest_f = float(pbest_f[g_i])

        history = [gbest_f]
        last_improve = 0
        n_turb = max(1, int(self.turbulence_frac * n))

        for it in range(max_iter):
            w = self.w_start + (self.w_end - self.w_start) * (it / max(1, max_iter - 1))

            attractor = (
                self._ring_best(pbest_x, pbest_f)
                if self.topology == "ring"
                else gbest_x[None, :]
            )

            r1 = rng.random((n, d))
            r2 = rng.random((n, d))
            v = (
                w * v
                + self.c1 * r1 * (pbest_x - x)
                + self.c2 * r2 * (attractor - x)
            )
            v = np.clip(v, -v_max, v_max)
            x = bounds.clip(x + v)

            f = func.batch(x)
            improved = f < pbest_f
            pbest_x[improved] = x[improved]
            pbest_f[improved] = f[improved]

            g_i = int(np.argmin(pbest_f))
            if pbest_f[g_i] < gbest_f - self.tol:
                gbest_f = float(pbest_f[g_i])
                gbest_x = pbest_x[g_i].copy()
                last_improve = it

            # Turbulent re-init of the worst particles after a stagnation window.
            if self.stagnation and it - last_improve >= self.stagnation:
                worst = np.argsort(pbest_f)[-n_turb:]
                x[worst] = bounds.sample(n_turb, rng)
                v[worst] = (rng.random((n_turb, d)) * 2 - 1) * v_max
                last_improve = it

            history.append(gbest_f)
            self._emit(callback, gbest_x, gbest_f, history, it + 1, self.name)

        converged = self.stagnation > 0 and (max_iter - 1 - last_improve) < self.stagnation
        meta = {"topology": self.topology, "n_particles": n}
        return gbest_x, gbest_f, history, converged, meta


class MultiSwarm(Optimizer):
    """Several swarms with different explore/exploit balances, sharing a best.

    This mirrors the multi-swarm setup used on the Delta traffic problem: an
    *exploratory* swarm (high inertia, high cognitive), a *balanced* swarm, and
    an *exploitative* swarm (low inertia, high social). Every ``migrate_every``
    iterations the swarms exchange their global best so good regions spread,
    while each keeps its own search character.
    """

    name = "MultiSwarm-PSO"

    _PRESETS = [
        dict(w_start=0.9, w_end=0.4, c1=2.5, c2=1.5),  # exploratory
        dict(w_start=0.7, w_end=0.4, c1=2.0, c2=2.0),  # balanced
        dict(w_start=0.5, w_end=0.2, c1=1.5, c2=2.5),  # exploitative
    ]

    def __init__(self, n_swarms: int = 3, n_particles: int = 20, migrate_every: int = 10):
        self.n_swarms = n_swarms
        self.n_particles = n_particles
        self.migrate_every = migrate_every

    def _run(self, func, bounds: Bounds, max_iter, rng, callback):
        d = bounds.dim
        v_max = 0.2 * bounds.span
        swarms = []
        for s in range(self.n_swarms):
            cfg = self._PRESETS[s % len(self._PRESETS)]
            x = bounds.sample(self.n_particles, rng)
            v = (rng.random((self.n_particles, d)) * 2 - 1) * v_max
            f = func.batch(x)
            swarms.append(
                dict(cfg=cfg, x=x, v=v, pbest_x=x.copy(), pbest_f=f.copy())
            )

        gbest_x = None
        gbest_f = np.inf
        for s in swarms:
            i = int(np.argmin(s["pbest_f"]))
            if s["pbest_f"][i] < gbest_f:
                gbest_f = float(s["pbest_f"][i])
                gbest_x = s["pbest_x"][i].copy()

        history = [gbest_f]
        for it in range(max_iter):
            frac = it / max(1, max_iter - 1)
            for s in swarms:
                cfg = s["cfg"]
                w = cfg["w_start"] + (cfg["w_end"] - cfg["w_start"]) * frac
                r1 = rng.random((self.n_particles, d))
                r2 = rng.random((self.n_particles, d))
                s["v"] = (
                    w * s["v"]
                    + cfg["c1"] * r1 * (s["pbest_x"] - s["x"])
                    + cfg["c2"] * r2 * (gbest_x[None, :] - s["x"])
                )
                s["v"] = np.clip(s["v"], -v_max, v_max)
                s["x"] = bounds.clip(s["x"] + s["v"])
                f = func.batch(s["x"])
                imp = f < s["pbest_f"]
                s["pbest_x"][imp] = s["x"][imp]
                s["pbest_f"][imp] = f[imp]
                i = int(np.argmin(s["pbest_f"]))
                if s["pbest_f"][i] < gbest_f:
                    gbest_f = float(s["pbest_f"][i])
                    gbest_x = s["pbest_x"][i].copy()

            # Migration: the worst swarm's worst particle adopts the global best.
            if self.migrate_every and (it + 1) % self.migrate_every == 0:
                worst_s = max(swarms, key=lambda s: s["pbest_f"].min())
                j = int(np.argmax(worst_s["pbest_f"]))
                worst_s["x"][j] = gbest_x.copy()

            history.append(gbest_f)
            self._emit(callback, gbest_x, gbest_f, history, it + 1, self.name)

        return gbest_x, gbest_f, history, False, {"n_swarms": self.n_swarms}


def pso(func, bounds: Bounds, max_iter: int = 200, seed: Optional[int] = None, **kw):
    """Functional shortcut: ``pso(f, bounds)`` -> :class:`Result`."""
    return ParticleSwarm(**kw).minimize(func, bounds, max_iter=max_iter, seed=seed)
