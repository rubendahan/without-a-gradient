# 03 — Benchmark functions

The library ships six standard global-optimization test landscapes in
`metaheuristics.benchmarks`. Each is a scalar function `f: ℝᵈ → ℝ` that works in
any dimension (except where noted), bundled with its metadata in a `Benchmark`
dataclass and registered in `BENCHMARKS`.

```python
from metaheuristics.benchmarks import rastrigin, get, BENCHMARKS

rastrigin([0.0, 0.0])           # -> 0.0
b = get("rastrigin")            # Benchmark(func, lower, upper, f_min, x_min, ...)
b.lower, b.upper, b.f_min       # recommended box and known optimum value
```

Why these six? Together they stress every weakness a metaheuristic can have:
convexity vs. ruggedness, separability vs. coupling, central vs. corner optima.

| Function | Box | f\* | Optimizer at | Modality | Stresses |
|---|---|---|---|---|---|
| `sphere` | ±5.12 | 0 | origin | unimodal | sanity check; convergence speed |
| `rosenbrock` | ±2.048 | 0 | (1,…,1) | unimodal* | ill-conditioning; following a curved valley |
| `rastrigin` | ±5.12 | 0 | origin | multimodal | ~10ᵈ regular local minima; escaping traps |
| `ackley` | ±32.768 | 0 | origin | multimodal | flat plateau + one deep funnel; needs global view |
| `griewank` | ±600 | 0 | origin | multimodal | coupled cosines; deceptive at small scale |
| `schwefel` | ±500 | 0 | (420.97,…) | multimodal | global optimum far from centre; anti-centre-bias |

\* Rosenbrock is technically unimodal in low dim but behaves like a hard
ill-conditioned valley; in `d ≥ 4` it grows a second local minimum.

---

## The landscapes, one by one

### `sphere(x) = Σ xᵢ²`
The convex bowl. Every solver must nail this; if it can't, something is broken.
Good for measuring raw convergence rate.

### `rosenbrock(x) = Σ [100(x_{i+1} − xᵢ²)² + (1 − xᵢ)²]`
The "banana valley": a narrow, curved, near-flat ridge. The minimum sits inside a
parabolic trough where the gradient is tiny along the valley floor. This is where
**CMA-ES** dominates (it learns the valley geometry) and where naive isotropic
methods crawl.

### `rastrigin(x) = 10d + Σ [xᵢ² − 10·cos(2π xᵢ)]`
A sphere wrapped in a cosine egg-carton — a regular grid of roughly `10ᵈ` local
minima of increasing depth toward the centre. The classic test of whether a method
can **escape local minima**. Population diversity (GA, PSO) tends to win.

### `ackley(x) = −20·exp(−0.2·√(Σxᵢ²/d)) − exp(Σcos(2πxᵢ)/d) + 20 + e`
A wide, almost-flat outer plateau with one deep funnel at the centre. Local
information on the plateau is nearly useless, so a method needs a **global view**
to find the funnel at all.

### `griewank(x) = 1 + Σxᵢ²/4000 − Π cos(xᵢ/√i)`
Product of cosines over a quadratic. Strongly multimodal at small scale but the
ripples flatten as the box grows, so difficulty paradoxically *drops* with
dimension — a nice reminder that "multimodal" is scale-dependent.

### `schwefel(x) = 418.9829·d − Σ xᵢ·sin(√|xᵢ|)`
**Deceptive on purpose:** the global optimum sits near a corner of the box
(`xᵢ ≈ 420.97`), far from the centre, while a strong second-best basin sits
elsewhere. Methods with any centre bias (or that converge before exploring the
periphery) get fooled — it's the hardest of the six for most optimizers.

---

## Adding your own

A benchmark is just a callable plus metadata:

```python
from metaheuristics.benchmarks import Benchmark, BENCHMARKS
import numpy as np

def levy(x):
    w = 1 + (np.asarray(x) - 1) / 4
    term1 = np.sin(np.pi * w[0]) ** 2
    term3 = (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
    term2 = np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2))
    return float(term1 + term2 + term3)

BENCHMARKS["levy"] = Benchmark(levy, -10, 10, 0.0, "(1,...,1)", True,
                               "Levy function: multimodal with a single global min.")
```

Then it's usable everywhere `compare()` and the examples are.
