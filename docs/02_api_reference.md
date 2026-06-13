# 02 — API reference

Everything is built on three objects and one method signature. Import the public
names straight from the top-level package:

```python
from metaheuristics import (
    Bounds, Result, Optimizer, as_minimizer,
    ParticleSwarm, MultiSwarm, GeneticAlgorithm, DifferentialEvolution,
    SimulatedAnnealing, HillClimbing, CMAES, BayesianOptimization,
    compare, run_trials, format_table,
)
from metaheuristics.benchmarks import rastrigin, ackley, get, BENCHMARKS
```

---

## The one method

Every optimizer exposes the same call:

```python
optimizer.minimize(func, bounds, max_iter=200, seed=None, callback=None) -> Result
```

| Arg | Meaning |
|---|---|
| `func` | objective `f(x: np.ndarray) -> float`, minimized |
| `bounds` | a `Bounds` defining the box search space |
| `max_iter` | number of iterations (generations / temperature steps / BO samples) |
| `seed` | int seed for full reproducibility |
| `callback` | optional `callback(Result)` fired every iteration (for live plots) |

Each algorithm also has a functional shortcut, e.g. `pso(func, bounds, ...)`,
`ga(...)`, `de(...)`, `sa(...)`, `cma_es(...)`, `bayes_opt(...)`,
`hill_climbing(...)`.

---

## `Bounds(lower, upper, dim=None)`

The box `[lower_i, upper_i]`. Scalars broadcast when `dim` is given.

```python
Bounds(-5.12, 5.12, dim=10)        # 10-D cube
Bounds([0, -1], [1, 1])            # per-axis limits
```

| Member | Description |
|---|---|
| `.dim` | dimensionality |
| `.span` | `upper - lower` per axis |
| `.clip(x)` | hard-project a point/batch into the box |
| `.reflect(x)` | reflect out-of-box coordinates back inside (gentler) |
| `.sample(n, rng)` | `n` uniform points → `(n, d)` array |

---

## `Result`

Uniform return value (a dataclass).

| Field | Description |
|---|---|
| `best_x` | best point found, `np.ndarray` |
| `best_f` | best objective value, `float` |
| `history` | best-so-far value per iteration (`list[float]`) |
| `n_evals` | total objective evaluations |
| `n_iter` | number of iterations recorded |
| `converged` | bool (algorithm-specific stagnation/precision flag) |
| `name` | optimizer name |
| `meta` | dict of extra algorithm-specific info |

---

## `Optimizer` (base class)

Abstract. Subclass and implement `_run(func, bounds, max_iter, rng, callback)`
returning `(best_x, best_f, history, converged, meta)`. The base class wraps the
objective in an evaluation counter, seeds a NumPy `Generator`, and assembles the
`Result`. You normally use the concrete optimizers below, not this directly.

`as_minimizer(func, maximize=False)` — returns `func`, or `-func` to maximize.

---

## Optimizer constructors

Only the most useful parameters are listed; see each module's docstring for the
full set and defaults.

### `ParticleSwarm`
```python
ParticleSwarm(n_particles=30, w_start=0.9, w_end=0.4, c1=2.05, c2=2.05,
              topology="global", ring_k=2, v_max_frac=0.2,
              stagnation=20, turbulence_frac=0.2, tol=1e-10)
```
`topology="ring"` follows local neighbours instead of the global best (slows
premature convergence). Inertia is annealed `w_start → w_end`.

### `MultiSwarm`
```python
MultiSwarm(n_swarms=3, n_particles=20, migrate_every=10)
```
Three swarms with exploratory / balanced / exploitative presets, sharing a global
best. The configuration used on Delta.

### `GeneticAlgorithm`
```python
GeneticAlgorithm(pop_size=60, tournament_k=3, crossover="blx",
                 crossover_rate=0.9, blx_alpha=0.5, mutation_rate=0.1,
                 sigma_frac=0.1, sigma_decay=0.99, elite=2)
```
`crossover` ∈ {`"uniform"`, `"blx"`}. Mutation sigma anneals by `sigma_decay`.

### `DifferentialEvolution`
```python
DifferentialEvolution(pop_size=40, F=0.7, CR=0.9, strategy="rand1")
```
`strategy` ∈ {`"rand1"` (robust), `"best1"` (greedier)}.

### `SimulatedAnnealing`
```python
SimulatedAnnealing(t_start=1.0, t_end=1e-3, step_frac=0.2, restart_after=50)
```
Geometric cooling calibrated to reach `t_end` by the last iteration. Proposal step
scales with `√(T/t_start)`.

### `HillClimbing`
```python
HillClimbing(step_frac=0.1, shrink=0.9, grow=1.1, restarts=1)
```
`restarts > 1` gives random-restart hill climbing.

### `CMAES`
```python
CMAES(pop_size=None, sigma0_frac=0.3)
```
`pop_size` defaults to Hansen's `4 + floor(3·ln d)`.

### `BayesianOptimization`
```python
BayesianOptimization(n_init=8, n_candidates=2000, length_scale_frac=0.2,
                     signal_var=1.0, noise=1e-4, xi=0.01)
```
Here `max_iter` is the number of *sequential surrogate-guided* evaluations after
the initial random design of `n_init` points. `xi` is the EI exploration margin.

---

## Comparison helpers — `compare.py`

```python
run_trials(optimizer, func, bounds, max_iter=200, seeds=range(10)) -> dict
compare(optimizers, func, bounds, max_iter=200, seeds=range(10)) -> list[dict]
format_table(rows, f_min=None) -> str
```

`run_trials` runs one optimizer over several seeds and returns
`{name, mean, std, best, worst, median, evals, curves}`. `compare` does it for a
list and sorts best-mean-first. `format_table` renders the rows as a monospace
table.

```python
rows = compare([ParticleSwarm(), DifferentialEvolution(), CMAES()],
               rastrigin, Bounds(-5.12, 5.12, dim=4), max_iter=120)
print(format_table(rows, f_min=0.0))
```

---

## Minimal end-to-end example

```python
import numpy as np
from metaheuristics import DifferentialEvolution, Bounds

def my_objective(x):           # any black box returning a float
    return np.sum((x - 0.3) ** 2) + 0.1 * np.sum(np.cos(20 * x))

bounds = Bounds(-1, 1, dim=8)
res = DifferentialEvolution(F=0.8, CR=0.9).minimize(
    my_objective, bounds, max_iter=150, seed=0,
    callback=lambda r: None,   # e.g. push r.best_f to a live plot
)
print(res.best_f, res.best_x)
```
