# Changelog

## 0.1.0 — 2026-06-13

Initial release. Extracted and cleaned up from the Delta 2026 hackathon
traffic-optimization project.

### Added
- **Optimizers** behind a unified `minimize(func, bounds) -> Result` API:
  - `ParticleSwarm`, `MultiSwarm` (PSO with inertia schedule, ring topology,
    turbulent re-init, multi-swarm migration)
  - `GeneticAlgorithm` (tournament selection, BLX-α / uniform crossover, Gaussian
    mutation with annealing, elitism)
  - `DifferentialEvolution` (rand/1/bin and best/1/bin, binomial crossover)
  - `SimulatedAnnealing` (geometric cooling, temperature-scaled proposals)
  - `HillClimbing` (adaptive step, random restarts)
  - `CMAES` (standard (μ/μ_w, λ) covariance matrix adaptation)
  - `BayesianOptimization` (GP surrogate with RBF kernel + Expected Improvement,
    NumPy-only)
- **Benchmarks**: `sphere`, `rosenbrock`, `rastrigin`, `ackley`, `griewank`,
  `schwefel`, with metadata registry.
- **Core**: `Bounds`, `Result`, `Optimizer` base class, evaluation counter.
- **Comparison harness**: `run_trials`, `compare`, `format_table`.
- **Tests**: 39 pytest tests (benchmark correctness, Result contract, determinism,
  progress, CMA-ES on Rosenbrock, callbacks).
- **Examples**: `compare_optimizers.py`, `plot_convergence.py`.
- **Interactive website** (`web/`): zero-build static explainer with live in-browser
  demos for PSO, GA, DE, SA, and Bayesian optimization.
- **Docs** (`docs/`): project story, algorithm theory, API reference, benchmark
  guide, reproducible results leaderboard.
