"""Benchmarks evaluate to their known global minimum at the known optimizer."""
import numpy as np
import pytest

from metaheuristics import benchmarks as bm


def test_sphere_minimum():
    assert bm.sphere(np.zeros(5)) == 0.0


def test_rosenbrock_minimum():
    assert bm.rosenbrock(np.ones(5)) == pytest.approx(0.0, abs=1e-12)


def test_rastrigin_minimum():
    assert bm.rastrigin(np.zeros(7)) == pytest.approx(0.0, abs=1e-12)


def test_ackley_minimum():
    assert bm.ackley(np.zeros(3)) == pytest.approx(0.0, abs=1e-10)


def test_griewank_minimum():
    assert bm.griewank(np.zeros(4)) == pytest.approx(0.0, abs=1e-12)


def test_schwefel_minimum():
    x = np.full(3, 420.9687)
    assert bm.schwefel(x) == pytest.approx(0.0, abs=1e-2)


@pytest.mark.parametrize("name", list(bm.BENCHMARKS))
def test_registry_metadata_consistent(name):
    b = bm.get(name)
    assert b.lower < b.upper
    # The function is callable and returns a finite scalar at the centre.
    val = b(np.zeros(2) + 0.5 * (b.lower + b.upper))
    assert np.isfinite(val)


def test_get_unknown_raises():
    with pytest.raises(KeyError):
        bm.get("does-not-exist")
