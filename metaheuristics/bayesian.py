"""Bayesian Optimization with a Gaussian-Process surrogate.

When each evaluation of ``f`` is *expensive* (a simulation, a training run, a lab
experiment), you cannot afford the thousands of samples a swarm or GA needs.
Bayesian optimization spends compute to *think* instead of to *sample*:

1. Fit a cheap probabilistic **surrogate** -- a Gaussian Process -- to the data
   seen so far. The GP returns, for any point, a predicted mean ``mu(x)`` and an
   uncertainty ``sigma(x)``.
2. Maximize an **acquisition function** over that surrogate to decide where to
   look next. We use *Expected Improvement*, which balances exploitation (low
   predicted ``mu``) against exploration (high ``sigma``):

       EI(x) = (f_best - mu(x)) * Phi(z)  +  sigma(x) * phi(z),
       z = (f_best - mu(x)) / sigma(x)

3. Evaluate the true ``f`` there, add the point, refit, repeat.

The result is remarkable sample efficiency: tens of evaluations where a population
method would use thousands. The cost is the ``O(n^3)`` GP fit, so BO is the tool
of choice precisely when ``f`` is so expensive that ``n`` stays small.

This is a compact, dependency-free GP (RBF kernel, fixed noise) with EI optimized
by sampling -- a faithful, readable reference rather than a production library.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .core import Bounds, Optimizer


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def _norm_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


class _GP:
    """Minimal zero-mean Gaussian Process with an isotropic RBF kernel."""

    def __init__(self, length_scale: float, signal_var: float, noise: float):
        self.l = length_scale
        self.sf = signal_var
        self.noise = noise

    def _kernel(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        d2 = (
            np.sum(A * A, 1)[:, None]
            + np.sum(B * B, 1)[None, :]
            - 2.0 * A @ B.T
        )
        return self.sf * np.exp(-0.5 * np.maximum(d2, 0.0) / (self.l ** 2))

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y_mean = y.mean()
        self.y = y - self.y_mean
        K = self._kernel(X, X) + self.noise * np.eye(len(X))
        self.L = np.linalg.cholesky(K + 1e-10 * np.eye(len(X)))
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y))
        return self

    def predict(self, Xs: np.ndarray):
        Ks = self._kernel(self.X, Xs)
        mu = Ks.T @ self.alpha + self.y_mean
        v = np.linalg.solve(self.L, Ks)
        var = self.sf - np.sum(v * v, 0)
        return mu, np.sqrt(np.maximum(var, 1e-12))


class BayesianOptimization(Optimizer):
    """GP-surrogate Bayesian optimization with Expected Improvement.

    Parameters
    ----------
    n_init:
        Random points sampled to seed the GP before EI takes over.
    n_candidates:
        Random candidates scored by EI each iteration to pick the next point
        (a simple, robust acquisition optimizer).
    length_scale_frac:
        RBF length scale as a fraction of the mean box span.
    noise:
        Observation noise variance (jitter); also regularizes the GP fit.
    xi:
        Exploration margin added inside EI; larger -> more exploratory.

    Notes
    -----
    Here ``max_iter`` is the number of *sequential, surrogate-guided* evaluations
    after the initial design -- BO's whole point is to make that number small.
    """

    name = "BayesOpt"

    def __init__(
        self,
        n_init: int = 8,
        n_candidates: int = 2000,
        length_scale_frac: float = 0.2,
        signal_var: float = 1.0,
        noise: float = 1e-4,
        xi: float = 0.01,
    ):
        self.n_init = n_init
        self.n_candidates = n_candidates
        self.length_scale_frac = length_scale_frac
        self.signal_var = signal_var
        self.noise = noise
        self.xi = xi

    def _run(self, func, bounds: Bounds, max_iter, rng, callback):
        X = bounds.sample(self.n_init, rng)
        y = func.batch(X)

        best_i = int(np.argmin(y))
        best_x, best_f = X[best_i].copy(), float(y[best_i])
        history = [best_f]

        length_scale = self.length_scale_frac * float(np.mean(bounds.span))
        gp = _GP(length_scale, self.signal_var, self.noise)

        for it in range(max_iter):
            # Standardize targets so the unit-variance GP kernel is well-scaled.
            y_std = y.std() or 1.0
            gp.fit(X, (y - y.mean()) / y_std)

            cand = bounds.sample(self.n_candidates, rng)
            mu, sigma = gp.predict(cand)
            mu = mu * y_std + y.mean()
            sigma = sigma * y_std

            f_best = y.min()
            z = (f_best - self.xi - mu) / np.maximum(sigma, 1e-12)
            ei = (f_best - self.xi - mu) * _norm_cdf(z) + sigma * _norm_pdf(z)
            ei[sigma < 1e-12] = 0.0

            x_next = cand[int(np.argmax(ei))]
            f_next = func(x_next)

            X = np.vstack([X, x_next])
            y = np.append(y, f_next)
            if f_next < best_f:
                best_f, best_x = float(f_next), x_next.copy()

            history.append(best_f)
            self._emit(callback, best_x, best_f, history, it + 1, self.name)

        return best_x, best_f, history, False, {"n_init": self.n_init}


def bayes_opt(func, bounds: Bounds, max_iter: int = 40, seed: Optional[int] = None, **kw):
    """Functional shortcut: ``bayes_opt(f, bounds)`` -> :class:`Result`."""
    return BayesianOptimization(**kw).minimize(func, bounds, max_iter=max_iter, seed=seed)
