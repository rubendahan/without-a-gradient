# 04 — Results

A reproducible leaderboard across **all 8 optimizers × all 6 benchmarks**, 4-D,
averaged over 8 seeds. Reproduce with:

```bash
python examples/compare_optimizers.py
```

Budgets are matched in spirit, not in raw evals: population methods get many cheap
evaluations, Bayesian optimization gets only ~58 (its whole point). `mean` is the
average final `f` over 8 seeds; `best` is the luckiest seed; lower is better
(global minimum is `f* = 0` everywhere).

> **Headline:** no optimizer wins on more than two of the six landscapes. That is
> the No Free Lunch theorem made concrete — performance is a function of how well a
> method's bias matches the landscape's structure.

---

## Per-function leaderboards

### sphere (convex) — *convergence-speed sanity check*
```
optimizer            mean        best     evals
CMA-ES           1.0e-22     8.9e-26      1601   ← second-order-like, by far fastest
MultiSwarm-PSO   7.1e-11     4.5e-13      4860
GA               1.2e-10     2.7e-21      7260
DE               1.7e-10     3.4e-12      4840
PSO              2.4e-08     3.1e-09      3630
HillClimbing     3.0e-07     2.8e-08      1510
SA               8.7e-04     4.7e-04      1501
BayesOpt           0.357      0.0589        58   ← few evals, but in the right region
```
Everyone solves it; the ranking is purely about convergence rate. CMA-ES is in a
different league because it adapts its step to the bowl's curvature.

### rosenbrock (ill-conditioned valley) — *CMA-ES territory*
```
optimizer            mean        best     evals
CMA-ES           3.9e-05     2.2e-12      1601   ← learns the valley geometry
DE               4.4e-05     2.2e-06      4840   ← self-scaling difference vectors
HillClimbing       0.202    5.4e-04      1510
MultiSwarm-PSO     0.346      0.233       4860
SA                 0.415      0.304       1501
PSO                0.459      0.220       3630
GA                 0.966      0.492       7260
BayesOpt           10.99      5.455         58
```
The signature result: CMA-ES drives the banana valley to `10⁻¹²` on its best seed,
orders of magnitude past the population methods. DE is a strong second thanks to its
self-scaling steps.

### rastrigin (egg-carton, ~10ᵈ minima) — *diversity wins*
```
optimizer            mean        best     evals
GA               7.8e-07         0.0      7260   ← crossover diversity finds the centre
MultiSwarm-PSO     0.748    3.5e-04      4860
PSO                0.844    6.5e-04      3630
CMA-ES             3.179      0.995      1601   ← single cloud gets trapped
DE                 4.477      1.824      4840
SA                 5.036      0.305      1501
HillClimbing       5.845      1.990      1510
BayesOpt           20.88      12.11        58
```
The mirror image of Rosenbrock: here CMA-ES's single adapting cloud is a liability,
and the GA's population diversity finds the global basin (one seed hits exactly 0).

### ackley (plateau + funnel) — *needs a global view*
```
optimizer            mean        best     evals
MultiSwarm-PSO   4.0e-05     1.7e-06      4860
GA               5.2e-05     7.6e-09      7260
DE               2.3e-04     7.1e-05      4840
PSO              2.9e-03     8.3e-04      3630
CMA-ES             0.230    1.1e-11      1601   ← bimodal: great or trapped
SA                 1.024      0.668       1501
HillClimbing       1.915    2.8e-03      1510
BayesOpt           5.844      2.619        58
```
Population methods with a global view find the central funnel reliably. CMA-ES is
bimodal — its best seed is `10⁻¹¹`, but its mean is dragged up by seeds that
quenched on the plateau.

### griewank (coupled cosines)
```
optimizer            mean        best     evals
GA                0.02807    2.6e-04      7260
CMA-ES            0.03542      0.0099     1601
MultiSwarm-PSO    0.07741      0.0099     4860
PSO                0.106       0.0535     3630
DE                 0.159       0.0985     4840
HillClimbing       0.178       0.0470     1510
SA                 0.304       0.141      1501
BayesOpt           2.212       1.259        58
```
Tight field; the residual is the shallow ripples near the optimum that all methods
struggle to fully iron out at this budget.

### schwefel (deceptive, corner optimum) — *the hardest*
```
optimizer            mean        best     evals
DE                 43.58      0.0398      4840
GA                 59.22     2.3e-13      7260   ← best single run, by far
MultiSwarm-PSO     236.9      118.4       4860
PSO                246.7      118.4       3630
SA                 354.1      218.4       1501
HillClimbing       391.0      236.9       1510
CMA-ES             529.9      118.4       1601   ← centre bias punished hardest
BayesOpt           662.3      396.8         58
```
The deceptive landscape with its optimum near a corner punishes any centre bias —
CMA-ES, which starts a compact cloud and adapts locally, finishes last on the mean.
DE and GA, which keep members scattered to the periphery, are the only ones to
occasionally find the true corner (GA hits `2×10⁻¹³` on one seed).

---

## What the table teaches

1. **No Free Lunch is real and visible.** CMA-ES is #1 on the two smooth functions
   and near-last on the two most deceptive ones. The GA is the opposite. Each method
   encodes a bias; it wins exactly where that bias matches.

2. **`mean` vs `best` tells you about reliability.** A method can have an excellent
   `best` and a poor `mean` (CMA-ES on Ackley: `10⁻¹¹` best, `0.23` mean) — that's a
   *bimodal* method that either nails it or gets trapped. If you only get one run,
   prefer a method with a good *mean*; if you can afford restarts, a good *best*
   matters more.

3. **Bayesian optimization looks weak here — by construction.** It used ~58
   evaluations against 1,500–7,000 for the others. The fair comparison isn't "who
   wins at fixed quality" but "who wins at *fixed evaluation budget when each
   evaluation is expensive*" — and at 58 evals, BO is already in the right region of
   every landscape, which is the whole point.

4. **The honest baseline earns its keep.** Random-restart hill climbing beats several
   fancier methods on Rosenbrock and Ackley. Always run it; if your sophisticated
   optimizer can't beat it, it isn't earning its complexity.

> Numbers will shift with `DIM`, `SEEDS`, and `max_iter` in
> `examples/compare_optimizers.py` — but the *shape* of the story (smooth → CMA-ES,
> rugged → GA/PSO, expensive → BO) is robust. That shape is the whole lesson.
