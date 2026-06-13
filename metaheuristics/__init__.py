"""metaheuristics -- a clean, unified library of black-box optimizers.

A small, dependency-light (NumPy-only) collection of population- and single-point
metaheuristics behind one consistent API. Every optimizer minimizes a scalar
objective over a box and returns the same :class:`Result`, so swapping algorithms
is a one-line change.

Quick start
-----------
>>> from metaheuristics import ParticleSwarm, Bounds
>>> from metaheuristics.benchmarks import rastrigin
>>> bounds = Bounds(-5.12, 5.12, dim=2)
>>> res = ParticleSwarm().minimize(rastrigin, bounds, seed=0)
>>> round(res.best_f, 3)
0.0

Available optimizers
--------------------
* :class:`ParticleSwarm`, :class:`MultiSwarm` -- particle swarm optimization
* :class:`GeneticAlgorithm`                   -- real-coded genetic algorithm
* :class:`DifferentialEvolution`              -- differential evolution
* :class:`SimulatedAnnealing`                 -- simulated annealing
* :class:`HillClimbing`                       -- (random-restart) hill climbing
* :class:`CMAES`                              -- covariance matrix adaptation ES
* :class:`BayesianOptimization`               -- GP-surrogate Bayesian optimization

Born from the Delta 2026 hackathon, where this whole toolbox was thrown at a
traffic-signal problem; cleaned up here into something reusable. See the project
story in ``docs/``.
"""
from .core import Bounds, Result, Optimizer, as_minimizer
from .pso import ParticleSwarm, MultiSwarm, pso
from .genetic import GeneticAlgorithm, ga
from .differential_evolution import DifferentialEvolution, de
from .simulated_annealing import SimulatedAnnealing, sa
from .hill_climbing import HillClimbing, hill_climbing
from .cma_es import CMAES, cma_es
from .bayesian import BayesianOptimization, bayes_opt
from .compare import compare, run_trials, format_table

__version__ = "0.1.0"

ALL_OPTIMIZERS = [
    ParticleSwarm,
    MultiSwarm,
    GeneticAlgorithm,
    DifferentialEvolution,
    SimulatedAnnealing,
    HillClimbing,
    CMAES,
    BayesianOptimization,
]

__all__ = [
    "Bounds", "Result", "Optimizer", "as_minimizer",
    "ParticleSwarm", "MultiSwarm", "pso",
    "GeneticAlgorithm", "ga",
    "DifferentialEvolution", "de",
    "SimulatedAnnealing", "sa",
    "HillClimbing", "hill_climbing",
    "CMAES", "cma_es",
    "BayesianOptimization", "bayes_opt",
    "compare", "run_trials", "format_table",
    "ALL_OPTIMIZERS", "__version__",
]
