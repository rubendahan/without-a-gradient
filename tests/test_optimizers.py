"""Every optimizer obeys the contract and makes real progress on benchmarks."""
import numpy as np
import pytest

from metaheuristics import (
    Bounds, ParticleSwarm, MultiSwarm, GeneticAlgorithm, DifferentialEvolution,
    SimulatedAnnealing, HillClimbing, CMAES, BayesianOptimization,
)
from metaheuristics.benchmarks import sphere, rastrigin, rosenbrock


ALL = [
    ParticleSwarm(n_particles=20),
    MultiSwarm(n_particles=12),
    GeneticAlgorithm(pop_size=30),
    DifferentialEvolution(pop_size=20),
    SimulatedAnnealing(),
    HillClimbing(restarts=4),
    CMAES(),
    BayesianOptimization(n_init=6),
]


@pytest.mark.parametrize("opt", ALL, ids=lambda o: o.name)
def test_result_contract(opt):
    """minimize returns a well-formed Result with a monotone best-so-far curve."""
    b = Bounds(-5.12, 5.12, dim=2)
    mi = 30 if opt.name == "BayesOpt" else 60
    res = opt.minimize(sphere, b, max_iter=mi, seed=0)

    assert res.best_x.shape == (2,)
    assert np.isfinite(res.best_f)
    assert res.n_evals > 0
    assert len(res.history) >= 1
    # Best-so-far must never increase.
    h = np.asarray(res.history)
    assert np.all(np.diff(h) <= 1e-9)
    # The reported best matches the end of the history curve.
    assert res.best_f == pytest.approx(h[-1], abs=1e-9)


@pytest.mark.parametrize("opt", ALL, ids=lambda o: o.name)
def test_makes_progress(opt):
    """Final value beats a random sample of the same budget, comfortably."""
    b = Bounds(-5.12, 5.12, dim=2)
    rng = np.random.default_rng(0)
    random_baseline = min(sphere(x) for x in b.sample(50, rng))

    mi = 30 if opt.name == "BayesOpt" else 80
    res = opt.minimize(sphere, b, max_iter=mi, seed=0)
    assert res.best_f < random_baseline


def test_determinism_same_seed():
    """Same seed -> identical result (population methods)."""
    b = Bounds(-5, 5, dim=3)
    a = ParticleSwarm().minimize(rastrigin, b, max_iter=40, seed=7)
    c = ParticleSwarm().minimize(rastrigin, b, max_iter=40, seed=7)
    assert a.best_f == c.best_f
    assert np.allclose(a.best_x, c.best_x)


def test_population_methods_solve_rastrigin_2d():
    """The global searchers should essentially nail Rastrigin in 2D."""
    b = Bounds(-5.12, 5.12, dim=2)
    for opt in [ParticleSwarm(), DifferentialEvolution(), GeneticAlgorithm()]:
        res = opt.minimize(rastrigin, b, max_iter=120, seed=0)
        assert res.best_f < 1e-2, opt.name


def test_cmaes_solves_rosenbrock():
    """CMA-ES's signature: learn the banana valley and drive it to ~0."""
    b = Bounds(-2.048, 2.048, dim=4)
    res = CMAES().minimize(rosenbrock, b, max_iter=250, seed=0)
    assert res.best_f < 1e-6


def test_callback_receives_updates():
    b = Bounds(-5, 5, dim=2)
    seen = []
    ParticleSwarm().minimize(
        sphere, b, max_iter=10, seed=0, callback=lambda r: seen.append(r.best_f)
    )
    assert len(seen) == 10
    assert seen == sorted(seen, reverse=True)  # best-so-far is non-increasing
