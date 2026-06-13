# 01 — Algorithms & theory

A written reference for every optimizer in the library. The
[interactive version](../web/index.html) shows each one running live; this is the
quieter, equation-first companion. The single thread running through all of them is
the trade-off between **exploration** (cover new ground) and **exploitation**
(refine what already looks good).

---

## The setup: black-box optimization

We want `x* = argmin f(x)` over a box `[lower, upper]^d`, where `f` is a **black
box**: you submit a candidate `x` and receive one number `f(x)`, with no gradient
and no analytic form. Two things make it hard:

- **Dimensionality** — search-space volume grows exponentially, so you can never
  grid it.
- **Multimodality** — the landscape has many local minima that masquerade as the
  global one.

No metaheuristic guarantees the global optimum in general. They are stochastic
sampling strategies that spend a fixed evaluation budget to find a *very good*
solution with high probability.

---

## 1. Particle Swarm Optimization — `pso.py`

A population of **particles**, each a candidate with a position `x` and velocity
`v`, flies through the space. Each remembers its personal best `p` and is drawn
toward the swarm's global best `g`. The update, applied every step:

```
v ← w·v + c1·r1·(p − x) + c2·r2·(g − x)      r1, r2 ~ U(0,1) per coordinate
x ← x + v
```

- **`w` (inertia)** — momentum. High `w` explores, low `w` exploits. Good PSO
  anneals `w` from ~0.9 to ~0.4 over the run.
- **`c1` (cognitive)** — pull toward personal memory.
- **`c2` (social)** — pull toward the swarm's best. High `c2` collapses the swarm
  onto the first decent point (fast, trap-prone).

**Failure mode:** premature convergence — once `g` stops improving, the difference
terms vanish and the swarm freezes. **Fixes in the library:** a *ring topology*
(follow a local neighbour's best so diversity survives) and *turbulent re-init* of
the worst particles after a stagnation window. `MultiSwarm` runs three swarms with
different explore/exploit balances and shares their best (the Delta setup).

**Reach for it when:** continuous, multimodal, you want few knobs and fast setup.

---

## 2. Genetic Algorithm — `genetic.py`

Candidate solutions are **individuals**; natural selection does the searching. Each
generation:

1. **Selection** — *tournament*: draw `k` random individuals, keep the best. Larger
   `k` = stronger selection pressure (faster, but diversity collapses sooner).
2. **Crossover** — recombine two parents. *BLX-α* samples each child gene from the
   interval spanned by the parents, extended by `α` so children can step slightly
   beyond either parent. (Holland's *building-block hypothesis*: good partial
   solutions combine into better whole ones.)
3. **Mutation** — Gaussian perturbation with an annealing step; the exploration
   valve.

**Elitism** copies the best few through untouched so the incumbent can't be lost.
The division of labour: crossover *exploits* by blending good blocks, mutation
*explores* by stepping off the population, selection supplies the pressure.

**GA vs PSO:** in PSO particles keep identity and are *attracted*; in a GA
individuals are *replaced* by recombinations. GAs shine on rugged/deceptive or
combinatorial structure where building blocks matter.

**Reach for it when:** rugged/multimodal or combinatorial problems; you can afford
to nurse diversity.

---

## 3. Differential Evolution — `differential_evolution.py`

The connoisseur's default for continuous problems. To perturb a target `xᵢ`, add
the **scaled difference of two other random members** (`rand/1/bin`):

```
donor v = x_a + F·(x_b − x_c)            a,b,c distinct random members
trial u = binomial-crossover(v, xᵢ, CR)  each coord from v w.p. CR, ≥1 guaranteed
xᵢ ← u   only if f(u) ≤ f(xᵢ)            (greedy selection)
```

- **`F` (differential weight, ~0.5–0.9)** — step aggressiveness.
- **`CR` (crossover prob, ~0.7–0.9)** — how much donor enters the trial. High `CR`
  suits separable functions; low `CR` suits tangled coordinates.

**The elegant part — self-scaling:** early, a spread-out population gives large
difference vectors → big exploratory steps; as it converges the differences shrink
→ fine local refinement, with *no step-size schedule*. `best/1/bin` biases the
donor toward the incumbent for greedier convergence.

**Reach for it when:** you want one robust, well-behaved default for a continuous
black box. Excellent on Rosenbrock-like valleys (second only to CMA-ES here).

---

## 4. Simulated Annealing — `simulated_annealing.py`

A single walker taking random steps, allowed to accept *worse* moves. Improvements
are always taken; a worsening move of size `Δ` is accepted with probability

```
P(accept) = exp(−Δ / T)
```

- **`T` (temperature)** — high `T` accepts almost anything (roams, escapes local
  minima); as `T → 0` only improvements survive (settles).
- **Cooling schedule** — geometric `T ← α·T`, `α` near 1. Too fast *quenches* into
  a bad minimum; too slow wastes the budget. This is the one parameter that matters.

The implementation scales the proposal step with `√(T/T₀)` and restarts the walker
to the best-known point after a streak of non-improving steps.

**Reach for it when:** cheap evaluations, you need to escape local minima, and you
don't want to maintain a population. It's the cheapest "escape local minima"
behaviour available — one point, one eval per step.

---

## 5. CMA-ES — `cma_es.py`

The Covariance Matrix Adaptation Evolution Strategy works on a *distribution*, not
on points. It maintains a multivariate Gaussian `N(m, σ²C)` and updates it each
generation:

1. **Sample** λ candidates from the Gaussian.
2. **Rank**, keep the best μ.
3. **Recombine** — new mean `m` = weighted average of the best μ.
4. **Adapt** `C` and `σ` so the Gaussian stretches along productive directions and
   shrinks along the rest, using two cumulative *evolution paths* that decouple
   step-size control from shape.

**Why it's special:** on an ill-conditioned curved valley (Rosenbrock), CMA-ES
learns the valley's orientation and scale — effectively recovering second-order
(Newton-like) information **without a gradient**.

**Trade-off:** the covariance update is `O(d²)` per step (an eigendecomposition),
so it's for **low-to-medium dimension** (`d ≲ 100`). On a regular multimodal grid
(Rastrigin) its single adapting cloud can still get trapped, where a diverse
population wins.

**Reach for it when:** smooth, ill-conditioned, modest-dimension problems where
sample efficiency matters.

---

## 6. Bayesian Optimization — `bayesian.py`

For when each `f(x)` is *expensive* (a simulation, a training run, a lab
experiment) and you can afford only tens of evaluations. Spend compute to *think*:

1. **Surrogate** — fit a *Gaussian Process* to the data so far. It returns, for any
   `x`, a predicted mean `μ(x)` and an honest uncertainty `σ(x)`.
2. **Acquisition** — *Expected Improvement* scores each candidate:
   ```
   EI(x) = (f_best − μ(x))·Φ(z) + σ(x)·φ(z),   z = (f_best − μ(x)) / σ(x)
   ```
   First term rewards low predicted mean (exploit); second rewards high uncertainty
   (explore). EI balances them automatically.
3. **Evaluate** `f` at the EI maximizer, add the point, refit, repeat.

**Trade-off:** dramatic sample efficiency, but the GP fit is `O(n³)`, so BO is for
the small-`n` regime. It's the standard engine behind automated hyperparameter
tuning. (Library implementation: RBF kernel, fixed noise, EI optimized by sampling,
NumPy-only — a readable reference, not a production GP.)

**Reach for it when:** evaluations are the bottleneck and you only get a few dozen.

---

## 7. Hill Climbing — `hill_climbing.py`

The honest baseline. From the current point take a small random step; keep it only
if it improves; shrink/grow the step adaptively. With **random restarts**, run many
short climbs from independent starts and return the best summit.

**Reach for it when:** you need a baseline every fancier method must beat. If a
swarm or GA can't beat random-restart hill climbing on your problem, it isn't
earning its complexity.

---

## Choosing — and No Free Lunch

The **No Free Lunch theorem** (Wolpert & Macready, 1997): averaged over *all*
objective functions, every optimizer performs identically. A method only wins by
matching your problem's structure. Practical decision tree:

- **Expensive evals?** → Bayesian optimization.
- **Smooth, ill-conditioned, `d ≲ 100`?** → CMA-ES.
- **Rugged multimodal continuous?** → DE / PSO / GA (and run random-restart hill
  climbing as the baseline).

And the biggest wins usually aren't the algorithm at all: a good **representation**,
a **warm start** from a heuristic, and a **fast surrogate** beat optimizer choice
almost every time. The optimizer is the last 10%.

See [`04_results.md`](04_results.md) for the empirical version of this story.
