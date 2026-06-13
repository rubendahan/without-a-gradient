# metaheuristics-lab

A clean, dependency-light (NumPy-only) library of **black-box metaheuristic
optimizers** behind one consistent API — plus a set of interactive web pages that
explain the **theory** of each algorithm with live, in-browser demos.

Every optimizer minimizes a scalar objective `f: ℝᵈ → ℝ` over a box and returns
the same `Result`, so swapping one algorithm for another is a one-line change.

```python
from metaheuristics import ParticleSwarm, GeneticAlgorithm, CMAES, Bounds
from metaheuristics.benchmarks import rastrigin

bounds = Bounds(-5.12, 5.12, dim=10)

res = ParticleSwarm().minimize(rastrigin, bounds, seed=0)
print(res)            # Result(name='PSO', best_f=..., best_x=[...], n_evals=...)

# Same call, different engine:
res = CMAES().minimize(rastrigin, bounds, seed=0)
```

> **Origin.** This toolbox was born at the **Delta 2026 hackathon**, where we
> threw every metaheuristic we knew at a city-scale traffic-signal problem.
> The contest task itself turned out to be near-trivial — but the optimizer
> zoo we built was worth keeping. So we cleaned it up, grounded it in standard
> benchmark functions, and wrote up the theory. See
> [`docs/00_project_story.md`](docs/00_project_story.md).

---

## What's inside

| Algorithm | Class | Family | Shines on |
|---|---|---|---|
| Particle Swarm Optimization | `ParticleSwarm`, `MultiSwarm` | swarm | multimodal, low effort |
| Genetic Algorithm | `GeneticAlgorithm` | evolutionary | rugged / multimodal |
| Differential Evolution | `DifferentialEvolution` | evolutionary | continuous, robust default |
| Simulated Annealing | `SimulatedAnnealing` | single-point | escaping local minima cheaply |
| Hill Climbing (+ restarts) | `HillClimbing` | single-point | the honest baseline |
| CMA-ES | `CMAES` | evolution strategy | ill-conditioned, smooth (`d ≲ 100`) |
| Bayesian Optimization | `BayesianOptimization` | surrogate | *expensive* objectives, few evals |

Six standard benchmark landscapes ship in `metaheuristics.benchmarks`:
`sphere`, `rosenbrock`, `rastrigin`, `ackley`, `griewank`, `schwefel`.

---

## Install

```bash
git clone https://github.com/rubendahan/metaheuristics-lab
cd metaheuristics-lab
pip install -e .          # add [dev] for pytest + matplotlib
```

Only runtime dependency: **NumPy**.

---

## Design in one minute

Three objects carry the whole library:

- **`Bounds(lower, upper, dim=None)`** — the box search space. Knows how to
  `sample`, `clip`, and `reflect` points.
- **`Result`** — uniform return value: `best_x`, `best_f`, `history`
  (best-so-far per iteration), `n_evals`, `n_iter`, `converged`, `meta`.
- **`Optimizer`** — abstract base. Each algorithm subclasses it and implements
  `_run`; the base class handles evaluation counting, seeding, and the `Result`.

```python
def minimize(self, func, bounds, max_iter=200, seed=None, callback=None) -> Result
```

A `callback(Result)` fires every iteration — handy for live plots or the web
demos. Maximize anything by wrapping it: `as_minimizer(f, maximize=True)`.

---

## Compare them yourself

```bash
python examples/compare_optimizers.py     # leaderboard on every benchmark
python examples/plot_convergence.py       # convergence curves (needs matplotlib)
```

The benchmark tells the textbook story cleanly (4-D, averaged over 8 seeds):

```
=== rosenbrock (4D) — the banana valley
optimizer             mean      best
CMA-ES            3.9e-05   2.2e-12     ← learns the valley geometry
DE                4.4e-05   2.2e-06
GA                 0.97      0.49

=== rastrigin (4D) — 10^d local minima
optimizer             mean      best
GA                7.8e-07      0.0      ← population diversity wins
PSO                0.84      6.5e-04
CMA-ES             3.18      0.99
```

> No optimizer wins everywhere — that's the **No Free Lunch** theorem in
> action. CMA-ES dominates smooth, ill-conditioned valleys; population methods
> (GA/PSO/DE) dominate rugged multimodal landscapes; Bayesian optimization wins
> when each evaluation is *expensive* and you only get a few dozen of them.

---

## The theory website

The `web/` folder is a static, zero-build **explorable explanation** of the
algorithms — open `web/index.html` in a browser, or host it on GitHub Pages.
Each page runs the actual algorithm live on a canvas: drag the inertia weight and
watch a swarm converge, cool a simulated-annealing walker, evolve a population,
or trace a Gaussian-process surrogate as Bayesian optimization picks its next
point.

---

## Tests

```bash
pip install -e ".[dev]"
pytest -q                 # 39 tests: benchmark correctness, the Result
                          # contract, determinism, and real progress
```

---

## License

MIT — see [`LICENSE`](LICENSE).
