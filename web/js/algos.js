// Live, canvas-drawable implementations of each metaheuristic for the demos.
// These mirror the Python package one-to-one but are written to be *watched*:
// one step() per animation frame, with draw() showing the search state.
import { makeMapper, rngFrom, gauss, TWO_PI } from "./landscape.js";

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

// ============================================================ PSO
export function createPSO(fn, p) {
  const rand = rngFrom(p.seed ?? 12345);
  const n = p.n | 0;
  const parts = [];
  for (let i = 0; i < n; i++) {
    const x = fn.lo + rand() * (fn.hi - fn.lo);
    const y = fn.lo + rand() * (fn.hi - fn.lo);
    parts.push({ x, y, vx: 0, vy: 0, px: x, py: y, pf: fn.f(x, y) });
  }
  let gx = parts[0].x, gy = parts[0].y, gf = Infinity;
  for (const pt of parts) if (pt.pf < gf) { gf = pt.pf; gx = pt.px; gy = pt.py; }

  return {
    iter: 0,
    get best() { return gf; },
    params: p,
    step() {
      this.iter++;
      const { w, c1, c2 } = this.params;
      const vmax = 0.25 * (fn.hi - fn.lo);
      for (const pt of parts) {
        const r1 = rand(), r2 = rand(), r3 = rand(), r4 = rand();
        pt.vx = w * pt.vx + c1 * r1 * (pt.px - pt.x) + c2 * r2 * (gx - pt.x);
        pt.vy = w * pt.vy + c1 * r3 * (pt.py - pt.y) + c2 * r4 * (gy - pt.y);
        pt.vx = clamp(pt.vx, -vmax, vmax);
        pt.vy = clamp(pt.vy, -vmax, vmax);
        pt.x = clamp(pt.x + pt.vx, fn.lo, fn.hi);
        pt.y = clamp(pt.y + pt.vy, fn.lo, fn.hi);
        const f = fn.f(pt.x, pt.y);
        if (f < pt.pf) { pt.pf = f; pt.px = pt.x; pt.py = pt.y; if (f < gf) { gf = f; gx = pt.x; gy = pt.y; } }
      }
    },
    draw(ctx, map) {
      for (const pt of parts) {
        const [sx, sy] = map.toPx(pt.x, pt.y);
        // velocity vector
        const [vx2, vy2] = map.toPx(pt.x + pt.vx * 2, pt.y + pt.vy * 2);
        ctx.strokeStyle = "rgba(88,166,255,0.45)";
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(vx2, vy2); ctx.stroke();
        ctx.fillStyle = "#58a6ff";
        ctx.beginPath(); ctx.arc(sx, sy, 3.2, 0, TWO_PI); ctx.fill();
      }
      drawBest(ctx, map, gx, gy);
    },
  };
}

// ============================================================ Genetic Algorithm
export function createGA(fn, p) {
  const rand = rngFrom(p.seed ?? 777);
  const n = p.n | 0;
  let pop = [];
  for (let i = 0; i < n; i++) {
    const x = fn.lo + rand() * (fn.hi - fn.lo);
    const y = fn.lo + rand() * (fn.hi - fn.lo);
    pop.push({ x, y, f: fn.f(x, y) });
  }
  let best = pop.reduce((a, b) => (a.f < b.f ? a : b));
  let lastChildren = [];

  const tournament = (k) => {
    let win = pop[(rand() * n) | 0];
    for (let j = 1; j < k; j++) { const c = pop[(rand() * n) | 0]; if (c.f < win.f) win = c; }
    return win;
  };

  return {
    iter: 0,
    get best() { return best.f; },
    params: p,
    step() {
      this.iter++;
      const { mut, k } = this.params;
      pop.sort((a, b) => a.f - b.f);
      const next = [{ ...pop[0] }, { ...pop[1] }]; // elitism
      lastChildren = [];
      const sigma = mut * (fn.hi - fn.lo);
      while (next.length < n) {
        const a = tournament(k), b = tournament(k);
        // BLX-0.5 crossover
        const blx = (u, v) => {
          const lo = Math.min(u, v), hi = Math.max(u, v), d = (hi - lo) * 0.5;
          return clamp(lo - d + rand() * (hi - lo + 2 * d), fn.lo, fn.hi);
        };
        for (let c = 0; c < 2 && next.length < n; c++) {
          let cx = blx(a.x, b.x), cy = blx(a.y, b.y);
          if (rand() < 0.5) cx = clamp(cx + gauss(rand) * sigma, fn.lo, fn.hi);
          if (rand() < 0.5) cy = clamp(cy + gauss(rand) * sigma, fn.lo, fn.hi);
          const child = { x: cx, y: cy, f: fn.f(cx, cy) };
          next.push(child); lastChildren.push(child);
          if (child.f < best.f) best = child;
        }
      }
      pop = next;
    },
    draw(ctx, map) {
      // colour by rank: brighter = fitter
      const sorted = [...pop].sort((a, b) => a.f - b.f);
      sorted.forEach((pt, i) => {
        const [sx, sy] = map.toPx(pt.x, pt.y);
        const t = i / sorted.length;
        ctx.fillStyle = `rgba(${86 + t * 120},${211 - t * 120},${99},0.9)`;
        ctx.beginPath(); ctx.arc(sx, sy, 3.4, 0, TWO_PI); ctx.fill();
      });
      drawBest(ctx, map, best.x, best.y);
    },
  };
}

// ============================================================ Differential Evolution
export function createDE(fn, p) {
  const rand = rngFrom(p.seed ?? 2024);
  const n = p.n | 0;
  const pop = [];
  for (let i = 0; i < n; i++) {
    const x = fn.lo + rand() * (fn.hi - fn.lo);
    const y = fn.lo + rand() * (fn.hi - fn.lo);
    pop.push({ x, y, f: fn.f(x, y) });
  }
  let bi = 0; for (let i = 1; i < n; i++) if (pop[i].f < pop[bi].f) bi = i;
  let trials = [];

  return {
    iter: 0,
    get best() { return pop[bi].f; },
    params: p,
    step() {
      this.iter++;
      const { F, CR } = this.params;
      trials = [];
      for (let i = 0; i < n; i++) {
        let a, b, c;
        do { a = (rand() * n) | 0; } while (a === i);
        do { b = (rand() * n) | 0; } while (b === i || b === a);
        do { c = (rand() * n) | 0; } while (c === i || c === a || c === b);
        const donx = clamp(pop[a].x + F * (pop[b].x - pop[c].x), fn.lo, fn.hi);
        const dony = clamp(pop[a].y + F * (pop[b].y - pop[c].y), fn.lo, fn.hi);
        const jx = rand() < CR || rand() < 0.5;
        const tx = jx ? donx : pop[i].x;
        const ty = !jx ? dony : (rand() < CR ? dony : pop[i].y);
        const tf = fn.f(tx, ty);
        trials.push({ x: tx, y: ty, i });
        if (tf <= pop[i].f) { pop[i] = { x: tx, y: ty, f: tf }; if (tf < pop[bi].f) bi = i; }
      }
    },
    draw(ctx, map) {
      for (const pt of pop) {
        const [sx, sy] = map.toPx(pt.x, pt.y);
        ctx.fillStyle = "#d2a8ff";
        ctx.beginPath(); ctx.arc(sx, sy, 3.2, 0, TWO_PI); ctx.fill();
      }
      drawBest(ctx, map, pop[bi].x, pop[bi].y);
    },
  };
}

// ============================================================ Simulated Annealing
export function createSA(fn, p) {
  const rand = rngFrom(p.seed ?? 99);
  let x = fn.lo + rand() * (fn.hi - fn.lo);
  let y = fn.lo + rand() * (fn.hi - fn.lo);
  let fx = fn.f(x, y);
  let bx = x, by = y, bf = fx;
  const trail = [[x, y, true]];
  let T = p.T0;

  return {
    iter: 0,
    get best() { return bf; },
    get temp() { return T; },
    params: p,
    step() {
      this.iter++;
      const { T0, cool } = this.params;
      const step = 0.18 * (fn.hi - fn.lo) * Math.sqrt(Math.max(T / T0, 1e-3));
      const nx = clamp(x + gauss(rand) * step, fn.lo, fn.hi);
      const ny = clamp(y + gauss(rand) * step, fn.lo, fn.hi);
      const nf = fn.f(nx, ny);
      const d = nf - fx;
      const accept = d < 0 || rand() < Math.exp(-d / Math.max(T, 1e-9));
      if (accept) { x = nx; y = ny; fx = nf; if (fx < bf) { bf = fx; bx = x; by = y; } }
      trail.push([x, y, accept]);
      if (trail.length > 120) trail.shift();
      T *= cool;
      if (T < T0 * 1e-4) T = T0 * 1e-4;
    },
    draw(ctx, map) {
      ctx.lineWidth = 1.4;
      for (let i = 1; i < trail.length; i++) {
        const [x0, y0] = trail[i - 1], [x1, y1, acc] = trail[i];
        const [a0, b0] = map.toPx(x0, y0), [a1, b1] = map.toPx(x1, y1);
        const alpha = i / trail.length;
        ctx.strokeStyle = acc ? `rgba(240,183,47,${0.25 + 0.6 * alpha})` : `rgba(255,123,114,${0.2 + 0.3 * alpha})`;
        ctx.beginPath(); ctx.moveTo(a0, b0); ctx.lineTo(a1, b1); ctx.stroke();
      }
      const [cx, cy] = map.toPx(x, y);
      ctx.fillStyle = "#f0b72f";
      ctx.beginPath(); ctx.arc(cx, cy, 4.5, 0, TWO_PI); ctx.fill();
      drawBest(ctx, map, bx, by);
    },
  };
}

function drawBest(ctx, map, x, y) {
  const [sx, sy] = map.toPx(x, y);
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(sx, sy, 6.5, 0, TWO_PI); ctx.stroke();
  ctx.fillStyle = "rgba(255,255,255,0.9)";
  ctx.beginPath(); ctx.arc(sx, sy, 2, 0, TWO_PI); ctx.fill();
}
