"""Benchmark every optimizer on every test function and print a leaderboard.

Run:  python examples/compare_optimizers.py
"""
from metaheuristics import (
    Bounds, ParticleSwarm, MultiSwarm, GeneticAlgorithm, DifferentialEvolution,
    SimulatedAnnealing, HillClimbing, CMAES, BayesianOptimization,
    compare, format_table,
)
from metaheuristics.benchmarks import BENCHMARKS

DIM = 4
SEEDS = range(8)


def optimizers():
    # Budgets are matched in spirit: population methods get many cheap evals,
    # Bayesian optimization gets few expensive ones (its whole point).
    return [
        (ParticleSwarm(), 120),
        (MultiSwarm(), 80),
        (GeneticAlgorithm(), 120),
        (DifferentialEvolution(), 120),
        (SimulatedAnnealing(), 1500),
        (HillClimbing(restarts=10), 1500),
        (CMAES(), 200),
        (BayesianOptimization(), 50),
    ]


def main():
    for name, bench in BENCHMARKS.items():
        bounds = Bounds(bench.lower, bench.upper, dim=DIM)
        print(f"\n=== {name}  ({DIM}D)  -- {bench.description}")
        rows = []
        for opt, max_iter in optimizers():
            rows.append(
                compare([opt], bench, bounds, max_iter=max_iter, seeds=SEEDS)[0]
            )
        rows.sort(key=lambda r: r["mean"])
        print(format_table(rows, f_min=bench.f_min))


if __name__ == "__main__":
    main()
