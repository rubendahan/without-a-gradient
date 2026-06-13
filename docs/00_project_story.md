# 00 — Project story

## The hackathon

This library was extracted from the **Delta 2026 hackathon**, Task D: a
traffic-signal optimization problem on a real city road network (Zagreb).

The task, in numbers:

| | |
|---|---|
| Road network | 1,717 intersections, 3,535 directed edges |
| Traffic lights to configure | **341** |
| Vehicles simulated | **5,215** |
| Simulation horizon | 14,400 s (4 hours), second-by-second mesoscopic sim |
| Objective | minimize the **total delay** (real time − free-flow time), summed over all vehicles |

Each traffic light is defined by **phases** (groups of approaches that get green
together), a **green duration** per phase (5–120 s, with 3 s of yellow between
phases), and an **offset** (the cycle's shift relative to *t = 0*). So the decision
vector lived in a few hundred dimensions, and the *only* way to score a candidate
was to run the entire 4-hour simulation. A textbook **expensive, high-dimensional,
black-box** optimization problem — exactly the regime metaheuristics exist for.

## What we threw at it

Pretty much the whole zoo, which is why this library exists:

- **Multi-swarm PSO** — three sub-swarms (exploratory / balanced / exploitative)
  with different inertia and cognitive/social weights, sharing a global best, run
  on a fast analytical *proxy* (~70 ms) instead of the full simulation (~5 s).
- **Genetic algorithms** — chromosomes over green durations, offsets, and source
  orderings; tournament selection, uniform crossover, targeted mutation, elitism.
- **Differential evolution** and **simulated annealing** on the proxy.
- **Bayesian optimization** — GP surrogate with Expected Improvement, plus random
  embeddings (REMBO) and trust-region (TuRBO-style) refinement to fight the
  dimensionality.
- A **Navier–Stokes fluid analogy** for green-time allocation (treat traffic as a
  compressible flow). Creative; not the winner.
- A **simulation-feedback loop** used as post-processing for every method: simulate
  → collect real per-signal arrival times → re-optimize each offset to catch
  arrivals on green → repeat to convergence.

The best result came from a multi-swarm PSO **seeded from a green-wave heuristic**
and polished by the feedback loop. Seeding from a good heuristic instead of random
init mattered far more than the choice of optimizer.

## The twist (honest postmortem)

Once we reverse-engineered the official simulator's exact scoring (it took several
diagnostic passes to nail the rounding and conflict rules), the problem turned out
to be nearly **trivial** for the given demand. The network was so under-saturated
that an "all-green, never stop anyone" configuration was already within a fraction
of a percent of optimal — there was almost no congestion for clever signal timing
to relieve. The elaborate optimizer tournament was, for *that* instance, overkill.

**The real lesson was the meta-lesson:** understand your problem before you
optimize it. We spent days tuning swarms when an afternoon of analyzing the demand
would have told us the ceiling was nearly flat.

## Why turn it into this

The contest answer was throwaway, but two things were worth keeping:

1. **The toolbox** — clean, reusable implementations of seven optimizers behind one
   API. Pulled out of the one-off hackathon scripts, rewritten against standard
   benchmark functions (where each algorithm's true character actually shows),
   documented, and tested.
2. **The understanding** — the theory of how each method balances exploration vs.
   exploitation, which is captured in the [interactive explainer](../web/index.html)
   and in [`01_algorithms.md`](01_algorithms.md).

That's this repository: a genuinely reusable package plus a teaching site, both
grounded in a real (if anticlimactic) optimization war story.

## Provenance

The original hackathon repository (not included here) contained ~40 solver scripts,
a reverse-engineered simulator, and a results archive. This library re-implements
the *general* algorithms cleanly and discards everything Delta-specific. Nothing
here depends on the traffic problem; it's all standard continuous black-box
optimization.
