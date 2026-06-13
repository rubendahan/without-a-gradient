# Build log — autonomous session

Record of what the background job built, for when you're back. Date: 2026-06-13.

## What you asked for

> From the Delta hackathon project, make (1) a clean, coherent, reusable codebase —
> GitHub-package style — and (2) a web page (like Paris Travel Time) explaining the
> theory we used: particle swarm, genetic algorithms, and the other techniques.
> Decoupled from the Delta problem itself (which turned out trivial — 0 red lights
> needed). Just show the theory we acquired. Document everything in markdown.

## What was built

A new standalone repo at `C:/Users/ruben/Documents/Projets/metaheuristics-lab`,
**independent of the Delta code** (nothing here imports or depends on it).

### 1. Python package `metaheuristics/` — reusable, NumPy-only

A clean library of 7 black-box optimizers behind **one unified API**
(`optimizer.minimize(func, bounds) -> Result`), grounded in standard benchmark
functions where each algorithm's real character shows:

| File | Contents |
|---|---|
| `core.py` | `Bounds`, `Result`, `Optimizer` base, eval counter |
| `benchmarks.py` | sphere, rosenbrock, rastrigin, ackley, griewank, schwefel |
| `pso.py` | `ParticleSwarm` (inertia schedule, ring topology, turbulence) + `MultiSwarm` |
| `genetic.py` | `GeneticAlgorithm` (tournament, BLX-α / uniform crossover, elitism) |
| `differential_evolution.py` | `DifferentialEvolution` (rand1/best1, binomial crossover) |
| `simulated_annealing.py` | `SimulatedAnnealing` (geometric cooling) |
| `hill_climbing.py` | `HillClimbing` (+ random restarts) — the baseline |
| `cma_es.py` | `CMAES` — full standard (μ/μ_w, λ) algorithm, ~90 lines |
| `bayesian.py` | `BayesianOptimization` — GP (RBF) + Expected Improvement, no scipy |
| `compare.py` | `run_trials`, `compare`, `format_table` |

Each module's top docstring **is** its theory section.

### 2. Tests, examples, packaging

- `tests/` — **39 pytest tests**, all passing: benchmark correctness, the `Result`
  contract (monotone history, eval counting), determinism, real progress vs. random
  baseline, CMA-ES on Rosenbrock, callback firing.
- `examples/compare_optimizers.py` — leaderboard across all optimizers × benchmarks.
- `examples/plot_convergence.py` — convergence curves (matplotlib).
- `pyproject.toml` (installable, `pip install -e .`), `LICENSE` (MIT), `.gitignore`,
  `README.md`.

### 3. Interactive theory website `web/` — static, zero-build, GitHub-Pages-ready

`web/index.html` + `styles.css` + `js/` — an explorable explanation with **live
canvas demos that run the real algorithms in-browser**:

- **PSO** — animated swarm + velocity vectors; sliders for inertia/cognitive/social,
  particle count, and landscape. Watch premature convergence happen live.
- **GA** — population colour-coded by fitness; sliders for population, mutation σ,
  tournament k. Watch the population "breathe" from exploration to exploitation.
- **DE** — candidate cloud self-scaling as it converges; F / CR sliders.
- **SA** — single walker with accepted/rejected trail and a live temperature readout;
  start-temp and cooling-α sliders. Watch it accept uphill moves while hot.
- **Bayesian optimization** — 1-D GP posterior (mean + ±2σ band) + Expected
  Improvement curve + next-sample marker; step through one expensive eval at a time.
- **CMA-ES** — theory section (no demo; a reshaping 2-D ellipse undersells it).
- A "which one do I use" decision table + No Free Lunch, and an honest **project
  story** section.

All five demos were syntax-checked with `node --check` and **logic-tested headless
with node** (they converge correctly: PSO/DE → 0, GA → 6e-4 on Rastrigin, Bayes
finds the 1-D optimum in ~15 evals).

### 4. Documentation `docs/`

- `00_project_story.md` — the Delta hackathon, what we threw at it, the honest "it
  was trivial" postmortem.
- `01_algorithms.md` — theory + update equations + tuning notes for every method.
- `02_api_reference.md` — `Bounds` / `Result` / `Optimizer` + every constructor.
- `03_benchmarks.md` — the six landscapes and what each one stresses.
- `04_results.md` — the reproducible leaderboard with analysis (No Free Lunch made
  concrete: smooth → CMA-ES, rugged → GA/PSO, expensive → BO).

## Verification done

- `pytest -q` → **39 passed**.
- `pip install -e .` → clean install.
- `python examples/compare_optimizers.py` → full leaderboard (captured in
  `docs/04_results.md`).
- `node --check` on all JS + headless run of all 5 demo algorithms → clean.
- CMA-ES verified on Rosenbrock 4-D → `1.8e-23` (correct: it learns the valley).

## Notes / decisions

- Built as a **new repo**, not edited in place over the Delta folder, because you
  wanted something reusable and decoupled. The Delta folder is untouched.
- The website is **vanilla JS, no build step** (unlike Paris Travel Time's
  React/Vite) so it hosts on GitHub Pages by just pushing `web/` — open
  `web/index.html` directly to preview.
- Bayesian optimization and CMA-ES are written as **readable references** (NumPy
  only, no scipy/cma dependency) — faithful to the standard algorithms but compact
  enough to teach from.

## Suggested next steps (not done — your call)

- `git init` + push to GitHub; enable Pages on `web/` (or move `web/` → `docs/` if
  you prefer the default Pages source). *(See "Git" below — I set up the repo and an
  initial commit; pushing is yours.)*
- Add a CMA-ES 2-D demo (animated covariance ellipse) if you want every algorithm
  visualized.
- Optionally add a small `pso`-on-the-traffic-proxy example to tie the library back
  to Delta, if you want the connection explicit.
