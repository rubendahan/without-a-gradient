"""Plot best-so-far convergence curves for several optimizers on one function.

Requires matplotlib (``pip install metaheuristics-lab[dev]``).
Run:  python examples/plot_convergence.py
"""
import numpy as np

from metaheuristics import (
    Bounds, ParticleSwarm, GeneticAlgorithm, DifferentialEvolution,
    SimulatedAnnealing, CMAES,
)
from metaheuristics.benchmarks import rastrigin


def main():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("matplotlib needed: pip install metaheuristics-lab[dev]")

    bounds = Bounds(-5.12, 5.12, dim=6)
    opts = [ParticleSwarm(), GeneticAlgorithm(), DifferentialEvolution(),
            SimulatedAnnealing(), CMAES()]

    plt.figure(figsize=(8, 5))
    for opt in opts:
        # Average the convergence curve over several seeds for a fair picture.
        curves = []
        for s in range(8):
            mi = 1500 if opt.name == "SA" else 150
            res = opt.minimize(rastrigin, bounds, max_iter=mi, seed=s)
            curves.append(res.history)
        L = min(len(c) for c in curves)
        mean_curve = np.mean([c[:L] for c in curves], axis=0)
        plt.plot(mean_curve, label=opt.name, linewidth=2)

    plt.yscale("log")
    plt.xlabel("iteration")
    plt.ylabel("best-so-far  f(x)   (log scale)")
    plt.title("Convergence on Rastrigin (6D), averaged over 8 seeds")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    out = "examples/convergence.png"
    plt.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
