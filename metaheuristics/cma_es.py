"""CMA-ES -- Covariance Matrix Adaptation Evolution Strategy.

CMA-ES is the heavyweight of black-box continuous optimization. Instead of moving
points around directly, it maintains a *distribution* -- a multivariate Gaussian
``N(m, sigma^2 C)`` -- and learns its shape from the best samples each generation:

1. **Sample** ``lambda`` candidates from the current Gaussian.
2. **Select** the best ``mu`` of them.
3. **Recombine** -- the new mean is their weighted average.
4. **Adapt** the covariance ``C`` and step size ``sigma`` so the distribution
   stretches along directions that have been paying off.

That covariance adaptation is what makes it special: on an ill-conditioned valley
like Rosenbrock it learns the valley's orientation and scale, effectively
performing a second-order search without ever computing a gradient. The trade-off
is the ``O(d^2)`` covariance update, so it shines in low-to-medium dimension
(roughly ``d <= 100``) where sample efficiency matters more than per-step cost.

This is a compact, faithful implementation of the standard (mu/mu_w, lambda)
algorithm from Hansen's CMA-ES tutorial -- enough to be genuinely useful and to
read as a teaching reference, without the production bells and whistles.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .core import Bounds, Optimizer


class CMAES(Optimizer):
    """Covariance Matrix Adaptation Evolution Strategy.

    Parameters
    ----------
    pop_size:
        Number of samples per generation (``lambda``). Default
        ``4 + floor(3*ln d)`` -- Hansen's recommendation.
    sigma0_frac:
        Initial step size as a fraction of the (uniform) box span.
    """

    name = "CMA-ES"

    def __init__(self, pop_size: Optional[int] = None, sigma0_frac: float = 0.3):
        self.pop_size = pop_size
        self.sigma0_frac = sigma0_frac

    def _run(self, func, bounds: Bounds, max_iter, rng, callback):
        n = bounds.dim
        lam = self.pop_size or (4 + int(3 * np.log(n)))
        mu = lam // 2

        # Recombination weights (log-decreasing), then the effective mass mu_eff.
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)

        # Adaptation rates (standard CMA-ES constants).
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        mean = bounds.sample(1, rng)[0]
        sigma = self.sigma0_frac * float(np.mean(bounds.span))
        C = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)

        best_x, best_f = mean.copy(), func(mean)
        history = [best_f]

        for it in range(max_iter):
            # Eigendecomposition gives B (axes) and D (per-axis std).
            C = np.triu(C) + np.triu(C, 1).T  # enforce symmetry
            eigval, B = np.linalg.eigh(C)
            eigval = np.clip(eigval, 1e-20, None)
            D = np.sqrt(eigval)

            z = rng.standard_normal((lam, n))
            y = z @ (B * D).T                 # y ~ N(0, C)
            X = mean + sigma * y
            X = bounds.clip(X)
            f = func.batch(X)

            order = np.argsort(f)
            if f[order[0]] < best_f:
                best_f = float(f[order[0]])
                best_x = X[order[0]].copy()

            y_sel = y[order[:mu]]
            y_w = w @ y_sel                   # weighted recombination step (in y-space)
            mean = mean + sigma * y_w
            mean = bounds.clip(mean)

            # Step-size control via the conjugate evolution path.
            C_inv_sqrt = B @ np.diag(1.0 / D) @ B.T
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(
                c_sigma * (2 - c_sigma) * mu_eff
            ) * (C_inv_sqrt @ y_w)
            sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chi_n - 1))

            # Covariance adaptation (rank-1 + rank-mu).
            hsig = (
                np.linalg.norm(p_sigma)
                / np.sqrt(1 - (1 - c_sigma) ** (2 * (it + 1)))
                / chi_n
            ) < (1.4 + 2 / (n + 1))
            p_c = (1 - c_c) * p_c + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w
            rank_mu = (y_sel * w[:, None]).T @ y_sel
            C = (
                (1 - c_1 - c_mu) * C
                + c_1 * (np.outer(p_c, p_c) + (1 - hsig) * c_c * (2 - c_c) * C)
                + c_mu * rank_mu
            )

            history.append(best_f)
            self._emit(callback, best_x, best_f, history, it + 1, self.name)

            if sigma < 1e-12:
                return best_x, best_f, history, True, {"sigma": sigma}

        return best_x, best_f, history, False, {"sigma": sigma}


def cma_es(func, bounds: Bounds, max_iter: int = 200, seed: Optional[int] = None, **kw):
    """Functional shortcut: ``cma_es(f, bounds)`` -> :class:`Result`."""
    return CMAES(**kw).minimize(func, bounds, max_iter=max_iter, seed=seed)
