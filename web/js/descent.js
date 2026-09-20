// The one-dimensional objective and the discrete gradient-descent trajectories.
export const DOMAIN = { lo: -4.2, hi: 5.2 };
export const STEP_SIZE = 0.025;
export const STARTS = Array.from(
  { length: 13 },
  (_, i) => DOMAIN.lo + 0.35 + ((i + 0.5) / 13) * (DOMAIN.hi - DOMAIN.lo - 0.7),
);
export function objective(x) {
  return (
    0.85 * Math.cos(1.7 * x) +
    0.5 * Math.cos(3.6 * x + 1.1) +
    0.32 * Math.cos(6.1 * x + 0.4) +
    0.09 * x * x -
    0.15 * x
  );
}
export function derivative(x) {
  const h = 1e-3;
  return (objective(x + h) - objective(x - h)) / (2 * h);
}
export function descend(start) {
  let x = Math.max(DOMAIN.lo, Math.min(DOMAIN.hi, start));
  const path = [x];
  for (let i = 0; i < 800; i++) {
    const g = derivative(x);
    if (Math.abs(g) < 1e-4) break;
    const next = Math.max(DOMAIN.lo, Math.min(DOMAIN.hi, x - STEP_SIZE * g));
    if (next === x) break;
    x = next;
    path.push(x);
  }
  return {
    start: path[0],
    path,
    rest: x,
    converged: Math.abs(derivative(x)) < 1e-4,
  };
}
// Numerical reference within the displayed interval, independent of the starts.
export const MINIMUM = (() => {
  let x = DOMAIN.lo;
  for (let i = 1; i <= 20000; i++) {
    const candidate = DOMAIN.lo + (i / 20000) * (DOMAIN.hi - DOMAIN.lo);
    if (objective(candidate) < objective(x)) x = candidate;
  }
  return { x, value: objective(x) };
})();

export function pointAt(run, iteration) {
  const i = Math.min(Math.floor(iteration), run.path.length - 1);
  const j = Math.min(i + 1, run.path.length - 1);
  const t = iteration - Math.floor(iteration);
  return run.path[i] + (run.path[j] - run.path[i]) * t;
}
