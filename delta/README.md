# Delta: traffic-signal retiming for the "Delta 2026" competition (Task D)

**A deliverable from our Delta 2026 competition team** (Delta was sponsored by
Mireo). This package contains the
optimisation approach we built for **Task D**, retiming a city's traffic signals
so a fleet of vehicles loses as little total time as possible over a 4-hour
simulation, together with the diagnostic that captures our key finding.

> ### ⚠️ Honest note on the simulator
> The real Delta objective is **Delta's proprietary mesoscopic traffic
> simulator**, which we do not have. So this package ships a transparent,
> self-contained **delay *proxy*** (`delta.simulator.DelayProxy`) that mirrors
> the real query interface. It is clearly marked as a stand-in everywhere.
> **To run on the real objective, drop the real simulator in by
> implementing one method**. See *[Plugging in the real simulator](#plugging-in-the-real-simulator)*.

---

## The problem (Task D)

The objective is **black-box**: you submit a full signal plan (a decision
vector) and the simulator runs the whole city and returns **one scalar**, the total
vehicle delay, lower is better. There is no gradient and no formula.

| Quantity | Real Delta scale |
|---|---|
| Signals (decision vector) | ~341 intersections |
| Vehicles simulated | ~5,215 |
| Simulated horizon | 14,400 s (4 h) per evaluation |
| Wall-clock per query | ~5 s |

Because each evaluation is slow, the **evaluation budget is the binding
constraint**. The decision variables per intersection are the signal-timing
parameters:

- **green split**: the fraction of the cycle each phase is green (splits sum to 1);
- **offset**: the cycle's shift vs. a global clock, used to coordinate adjacent
  intersections into a "green wave";
- optionally **cycle length**.

Bounds are physical (minimum green per phase, offset in `[0, cycle)`, cycle in a
legal range).

## Our method

We treated this as expensive black-box optimisation and threw our
**metaheuristics-lab** library at it:

- **Multi-population PSO** as the primary optimiser, with three sub-swarms
  (exploratory / balanced / exploitative) sharing a global best
  (`metaheuristics.MultiSwarm`).
- A **genetic algorithm** over the signal timings.
- **Differential evolution** and **simulated annealing** run on a *fast
  analytical proxy* of the simulator, so we could afford many evaluations.
- **Bayesian optimisation with random embeddings** to fight the 341-dimensionality.

The optimisers all come from our separate library via the unified
`optimizer.minimize(func, bounds) -> Result` API, so swapping algorithms is a
one-line change. This package is just the **problem encoding + glue + diagnostics**.

## The finding (the twist)

> **Characterise the objective before optimising it.**

Once we reverse-engineered the scoring, the network turned out to be so
**under-saturated** for the given demand that **almost any sane plan was within
<1% of optimal**. A demand-proportional "fair share of green, never needlessly
stop anyone" plan, which is Webster's textbook rule of thumb and takes five
minutes to write down with *no optimisation at all*, was essentially the answer. Our fleet of
metaheuristics could not meaningfully beat it.

This package demonstrates that flat ceiling on the proxy. `delta.analysis.characterize`
compares the sane plan against an optimised plan across a sweep of network loads:

```
  load  mean sat      sane plan      optimised   gain %  verdict
----------------------------------------------------------------
  0.40     0.325      2537381.9      2534163.3    0.127  flat ceiling
  0.55     0.446      3931705.6      3928465.3    0.082  flat ceiling
  0.70     0.568      5805475.2      5818470.3   -0.224  flat ceiling
  0.95     0.771     13247985.9     13382617.2   -1.016  flat ceiling
  1.10     0.893     98881288.4     96407621.6    2.502  TIMING MATTERS
  1.30     1.055    556380556.2    536622277.0    3.551  TIMING MATTERS
```

(`mean sat` is the mean degree of saturation `x = q/(s·g)`; `gain %` is how much
the optimiser improves on the sane plan.) While the network is undersaturated
the gain is ~0, and the optimiser earns nothing. Only once demand is pushed to/over
capacity (`mean sat → 1`) does the delay explode and timing start to matter.
**The real Delta network lived in the flat regime**, which is exactly why the
valuable deliverable was characterising the objective, not out-optimising it.

## How to run

Requires Python ≥ 3.9, NumPy, and our `metaheuristics-lab` library.

```bash
# 1. install the optimiser library (the parent repo), then this package
pip install -e ..
pip install -e .

# 2. end-to-end demo: build a city, optimise it, run the diagnostic
python -m delta

# 3. the two example scripts
python examples/optimize_city.py     # PSO vs. baselines, with decoded timings
python examples/flat_ceiling.py      # reproduce the flat-ceiling sweep above

# 4. tests
python -m pytest
```

The example scripts also add the project root to `sys.path`, so they run from a
fresh checkout without installing.

## Plugging in the real simulator

Everything downstream depends on a single method:

```python
def evaluate(self, plan_vector: numpy.ndarray) -> float:
    """Run the city for the full horizon and return total vehicle delay
    in vehicle-seconds (lower is better)."""
```

`plan_vector` is the unit-cube decision vector produced by `delta.plan.SignalPlan`.
Decode it with `plan.from_vector(plan_vector)` to recover per-intersection green
splits, offsets and cycle lengths (exactly as `DelayProxy` does). Then:

```python
from delta import build_example_city, solve   # or your own RoadNetwork
from delta.plan import SignalPlan

class DeltaSimulator:
    def evaluate(self, plan_vector):
        timings = plan.from_vector(plan_vector)   # decode if you need structure
        return run_delta_mesoscopic_sim(timings)  # the real engine -> total delay

network = build_example_city()                    # or describe your real city
plan = SignalPlan(network)
outcome = solve(network, simulator=DeltaSimulator())
print(outcome.result.best_f, outcome.result.best_x)
```

Nothing else changes: the encoder, the multi-population PSO and the
flat-ceiling diagnostic all run unmodified on the real objective. See the
`SIMULATOR-SWAP` note in `delta/simulator.py`.

## The proxy delay model (what the stand-in computes)

`DelayProxy` is a classic **Webster / HCM-style intersection delay** model,
NumPy-only and fully documented in `delta/simulator.py`. Per movement it sums:

1. **Uniform delay**: the unavoidable wait on red under steady demand,
   `0.5·C·(1−g)² / (1 − x·g)`.
2. **Overflow / random delay**: the HCM incremental term that is ~0 while the
   movement is below capacity and **explodes as the degree of saturation
   `x → 1`**. This is what makes the landscape flat-then-steep.

Plus a small **corridor-coordination** term that rewards offsets forming a good
green wave. The proxy reproduces the *qualitative* behaviour that drove our
finding (flat and cheap while undersaturated, steep near capacity) on a model we
can fully explain. It is **not** Delta's physics; it is an honest stand-in with
a matching interface.

## Package layout

```
delta-mireo/
  delta/
    network.py     # data model: Movement / Phase / Intersection / RoadNetwork; example city
    plan.py        # SignalPlan: decision-vector <-> structured timings, bounds, baselines
    simulator.py   # DelayProxy (the stand-in) + Simulator interface (the swap point)
    solver.py      # solve(): wires metaheuristics-lab (multi-population PSO) to the problem
    analysis.py    # characterize(): the flat-ceiling diagnostic
    __main__.py    # `python -m delta`
  examples/
    optimize_city.py
    flat_ceiling.py
  tests/
    test_plan.py test_simulator.py test_solver.py
  pyproject.toml LICENSE README.md
```

## License

MIT. See [LICENSE](LICENSE).
