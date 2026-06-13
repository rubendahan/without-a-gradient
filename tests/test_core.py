"""Core abstractions: Bounds geometry and Result bookkeeping."""
import numpy as np
import pytest

from metaheuristics import Bounds, as_minimizer
from metaheuristics.core import _EvalCounter


def test_bounds_broadcast_scalar():
    b = Bounds(-1, 1, dim=4)
    assert b.dim == 4
    assert np.allclose(b.span, 2.0)


def test_bounds_clip_and_sample():
    b = Bounds([0, 0], [1, 1])
    pt = b.clip(np.array([5.0, -3.0]))
    assert np.allclose(pt, [1.0, 0.0])
    rng = np.random.default_rng(0)
    pts = b.sample(100, rng)
    assert pts.shape == (100, 2)
    assert pts.min() >= 0.0 and pts.max() <= 1.0


def test_bounds_reflect_stays_inside():
    b = Bounds([-2, -2], [2, 2])
    rng = np.random.default_rng(1)
    far = rng.normal(0, 10, size=(500, 2))
    refl = b.reflect(far)
    assert refl.min() >= -2 - 1e-9 and refl.max() <= 2 + 1e-9


def test_bounds_rejects_inverted():
    with pytest.raises(ValueError):
        Bounds([1], [0])


def test_eval_counter_tracks_best():
    c = _EvalCounter(lambda x: float(np.sum(x ** 2)))
    c(np.array([3.0]))
    c(np.array([1.0]))
    c(np.array([2.0]))
    assert c.n_evals == 3
    assert c.best_f == 1.0
    assert np.allclose(c.best_x, [1.0])


def test_as_minimizer_negates():
    f = lambda x: x[0]
    g = as_minimizer(f, maximize=True)
    assert g(np.array([5.0])) == -5.0
