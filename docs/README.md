# Documentation

Reference material for **metaheuristics-lab**. The interactive, visual version of
the theory lives in [`../web/index.html`](../web/index.html); these pages are the
written reference.

| Doc | What's in it |
|---|---|
| [00 — Project story](00_project_story.md) | Where the library came from: the Delta 2026 hackathon, what we built, and the honest postmortem. |
| [01 — Algorithms & theory](01_algorithms.md) | The theory behind every optimizer (PSO, GA, DE, SA, CMA-ES, Bayesian, hill climbing), with the update equations and tuning notes. |
| [02 — API reference](02_api_reference.md) | `Bounds`, `Result`, `Optimizer`, and every optimizer's constructor parameters. |
| [03 — Benchmarks](03_benchmarks.md) | The six standard test landscapes and what each one stresses. |
| [04 — Results](04_results.md) | Reproducible leaderboard across all optimizers × all benchmarks, with discussion. |

## Quick map of the code

```
metaheuristics/
├── core.py                  Bounds, Result, Optimizer base, eval counter
├── benchmarks.py            sphere, rosenbrock, rastrigin, ackley, griewank, schwefel
├── pso.py                   ParticleSwarm, MultiSwarm
├── genetic.py               GeneticAlgorithm
├── differential_evolution.py DifferentialEvolution
├── simulated_annealing.py   SimulatedAnnealing
├── hill_climbing.py         HillClimbing (+ random restarts)
├── cma_es.py                CMAES
├── bayesian.py              BayesianOptimization (GP + Expected Improvement)
└── compare.py               run_trials, compare, format_table
web/                         the interactive explainer (static, no build)
examples/                    compare_optimizers.py, plot_convergence.py
tests/                       pytest suite (39 tests)
```

Each algorithm module's top-of-file docstring **is** its theory section — reading
the source is meant to teach, not just run.
