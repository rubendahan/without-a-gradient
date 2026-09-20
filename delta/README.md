# Traffic-signal optimisation

A signal-timing model and particle swarm application developed from the
Delta 2026 competition.

[Open the traffic simulation](https://rubendahan.github.io/without-a-gradient/delta/)

## Browser experiment

The page uses a synthetic grid of 36 junctions. Each junction has two green-time
weights and an offset, giving 108 search parameters. The cycle length is fixed.

Compare three plans:

- **Equal split**: half the cycle for each phase, with zero offsets.
- **Demand-based**: green allocation follows the critical flow ratios, with a
  minimum green time for each phase.
- **Optimised**: the best plan found by three groups of 16 particles, starting
  with the demand-based plan included in the population.

A run lasts 60 generations, or 2,928 objective evaluations including
initialisation. The seed is fixed. Pause and resume preserve the run; changing
the demand starts a new experiment.

## Delay model

The objective estimates total vehicle delay over four hours. It sums uniform
red-light delay, an incremental delay term and a heuristic corridor-coordination
term. Lower values are better.

The formulation follows a Webster / HCM-style intersection model. It is not
calibrated to a real city and does not include turning traffic, queue spillback
or clearance intervals. The animated cars are a separate illustration of signal
timing, not the source of the reported delay.

The official competition simulator is not included. Results from this model do
not reproduce competition scores or establish a real network's optimum.

## Run the page

From this directory, with Node.js 22.12 or later:

```bash
cd web
npm ci
npm run dev -- --port 5174
```

Open <http://localhost:5174/without-a-gradient/delta/>. Run `npm run build` for a
production build and `npm run lint` for the frontend checks.

## Python package

From the repository root:

```bash
python -m pip install -e . -e "./delta[dev]"
cd delta
python -m delta
python -m pytest
```

The Python package provides the network representation, signal-plan encoding,
delay model and solver. The browser has its own TypeScript implementations.

```python
from delta import build_example_city, solve

network = build_example_city()
outcome = solve(network, max_iter=40, seed=0)
print(outcome.result.best_f)
```

To use another objective, pass an object with an
`evaluate(plan_vector) -> float` method to `solve(network, simulator=...)`.
`SignalPlan.from_vector` decodes the normalised search vector into timings.
An external simulator adapter must translate those timings into its own network
and phase representation.

## Files

- `delta/network.py`: network data structures and example network.
- `delta/plan.py`: signal timings, bounds and baseline plans.
- `delta/simulator.py`: analytical delay model and simulator interface.
- `delta/solver.py`: optimisation through the parent NumPy library.
- `delta/analysis.py`: optional command-line comparisons across demand levels.
- `web/src/`: React interface and TypeScript model.
- `tests/`: Python model and solver tests.

[MIT license](LICENSE).
