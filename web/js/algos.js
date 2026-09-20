// The metaheuristics, instrumented for slow, legible animation. Each exposes:
//   step()   advance one discrete iteration
//   frame()  fully-resolved geometry {dots, rings, links} in domain coords,
//            with the defining OPERATOR drawn explicitly (the forces, the
//            difference vector, the crossover, the Metropolis move)
//   info()   live scalar state for the maths panel  [[label, value], ...]
//   status() one short sentence of narration
// The harness tweens between successive frame()s, so motion stays smooth.
import { rngFrom, gauss } from "./landscape.js";

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const COL = {
  agent: "#2864a5",
  de: "#79529a",
  best: "#223447",
  cog: "#278362",
  soc: "#2864a5",
  diff: "#8760a2",
  warm: "#ae7024",
  reject: "#c35938",
  cma: "#287f85",
};

// Eigendecomposition of a 2x2 symmetric matrix [[a,b],[b,d]]: the two
// eigenvalues and an orthonormal pair of eigenvectors. CMA-ES needs this to
// sample from N(0, C) and to form C^{-1/2}.
function eig2(a, b, d) {
  const tr = a + d,
    t2 = (a - d) / 2,
    disc = Math.sqrt(t2 * t2 + b * b);
  const l1 = tr / 2 + disc,
    l2 = tr / 2 - disc;
  let v1x, v1y;
  if (Math.abs(b) > 1e-12) {
    v1x = l1 - d;
    v1y = b;
  } else if (a >= d) {
    v1x = 1;
    v1y = 0;
  } else {
    v1x = 0;
    v1y = 1;
  }
  const nrm = Math.hypot(v1x, v1y) || 1;
  v1x /= nrm;
  v1y /= nrm;
  return {
    l1: Math.max(l1, 1e-20),
    l2: Math.max(l2, 1e-20),
    v1x,
    v1y,
    v2x: -v1y,
    v2y: v1x,
  };
}

function spread(pts, fn) {
  let mx = 0,
    my = 0;
  for (const p of pts) {
    mx += p.x;
    my += p.y;
  }
  mx /= pts.length;
  my /= pts.length;
  let s = 0;
  for (const p of pts) s += Math.hypot(p.x - mx, p.y - my);
  return s / pts.length / (fn.hi - fn.lo);
}

// ============================================================ PSO
export function createPSO(fn, p) {
  const rand = rngFrom(p.seed ?? 12345);
  const n = p.n | 0,
    parts = [];
  for (let i = 0; i < n; i++) {
    const x = fn.lo + rand() * (fn.hi - fn.lo),
      y = fn.lo + rand() * (fn.hi - fn.lo);
    parts.push({ x, y, vx: 0, vy: 0, px: x, py: y, pf: fn.f(x, y) });
  }
  let gx = parts[0].x,
    gy = parts[0].y,
    gf = Infinity;
  let operation = null;
  for (const pt of parts)
    if (pt.pf < gf) {
      gf = pt.pf;
      gx = pt.px;
      gy = pt.py;
    }

  return {
    iter: 0,
    params: p,
    stepsPerTick: 1,
    get best() {
      return gf;
    },
    status() {
      const s = spread(parts, fn);
      if (this.iter === 0)
        return "Initialised: positions xᵢ ~ U(box), velocities vᵢ = 0.";
      if (s > 0.25) return "The particles are spread across the search domain.";
      if (s > 0.09)
        return "The particle cloud is more concentrated. Compare its spread with the best-value curve.";
      return "The particles are close together. This alone does not establish convergence to an optimum.";
    },
    step() {
      this.iter++;
      const { w, c1, c2 } = this.params,
        vmax = 0.3 * (fn.hi - fn.lo);
      for (const pt of parts) {
        if (pt === parts[0])
          operation = { x: pt.x, y: pt.y, px: pt.px, py: pt.py, gx, gy };
        pt.vx =
          w * pt.vx + c1 * rand() * (pt.px - pt.x) + c2 * rand() * (gx - pt.x);
        pt.vy =
          w * pt.vy + c1 * rand() * (pt.py - pt.y) + c2 * rand() * (gy - pt.y);
        pt.vx = clamp(pt.vx, -vmax, vmax);
        pt.vy = clamp(pt.vy, -vmax, vmax);
        pt.x = clamp(pt.x + pt.vx, fn.lo, fn.hi);
        pt.y = clamp(pt.y + pt.vy, fn.lo, fn.hi);
        if (pt === parts[0]) {
          operation.nx = pt.x;
          operation.ny = pt.y;
        }
        const f = fn.f(pt.x, pt.y);
        if (f < pt.pf) {
          pt.pf = f;
          pt.px = pt.x;
          pt.py = pt.y;
          if (f < gf) {
            gf = f;
            gx = pt.x;
            gy = pt.y;
          }
        }
      }
    },
    info() {
      return [
        ["best f(ĝ)", gf.toExponential(2)],
        ["w", (+this.params.w).toFixed(2)],
        [
          "c₁ / c₂",
          `${(+this.params.c1).toFixed(1)} / ${(+this.params.c2).toFixed(1)}`,
        ],
        ["swarm spread", spread(parts, fn).toFixed(3)],
      ];
    },
    frame() {
      const h = parts[0];
      // faint velocity field vᵢ on every particle (one step ahead), then the
      // two named forces drawn as bold arrows on the highlighted particle.
      const vel = parts.map((pt) => ({
        x1: pt.x,
        y1: pt.y,
        x2: pt.x + pt.vx,
        y2: pt.y + pt.vy,
        color: "#658bb4",
        width: 1,
        alpha: 0.3,
        vel: true,
      }));
      return {
        operation,
        dots: parts.map((pt) => ({
          x: pt.x,
          y: pt.y,
          r: 3.6,
          color: COL.agent,
        })),
        links: [
          ...vel,
          {
            x1: h.x,
            y1: h.y,
            x2: h.px,
            y2: h.py,
            color: COL.cog,
            width: 2.2,
            arrow: true,
            glow: true,
            label: "c₁ → pᵢ (cognitive)",
          },
          {
            x1: h.x,
            y1: h.y,
            x2: gx,
            y2: gy,
            color: COL.soc,
            width: 2.2,
            arrow: true,
            glow: true,
            label: "c₂ → ĝ (social)",
          },
        ],
        rings: [
          { x: h.x, y: h.y, r: 6, color: COL.warm, label: "particle xᵢ" },
          { x: gx, y: gy, r: 7, color: COL.best, label: "ĝ best" },
        ],
      };
    },
  };
}

// ============================================================ Genetic Algorithm
export function createGA(fn, p) {
  const rand = rngFrom(p.seed ?? 777);
  const n = p.n | 0;
  let pop = [];
  for (let i = 0; i < n; i++) {
    const x = fn.lo + rand() * (fn.hi - fn.lo),
      y = fn.lo + rand() * (fn.hi - fn.lo);
    pop.push({ x, y, f: fn.f(x, y) });
  }
  let best = pop.reduce((a, b) => (a.f < b.f ? a : b));
  let mating = null; // {p1, p2, child} for the operator highlight

  const tour = (k) => {
    let w = pop[(rand() * n) | 0];
    for (let j = 1; j < k; j++) {
      const c = pop[(rand() * n) | 0];
      if (c.f < w.f) w = c;
    }
    return w;
  };

  return {
    iter: 0,
    params: p,
    stepsPerTick: 1,
    get best() {
      return best.f;
    },
    status() {
      const s = spread(pop, fn);
      if (this.iter === 0) return "Generation 0: a uniform random population.";
      if (s > 0.22)
        return "Tournament selection + BLX-α crossover; population still diverse.";
      if (s > 0.07) return "Selection pressure concentrates the gene pool.";
      return "The population has low spread. Elitism preserves the best candidate.";
    },
    step() {
      this.iter++;
      const { mut, k } = this.params;
      pop.sort((a, b) => a.f - b.f);
      const next = [{ ...pop[0] }, { ...pop[1] }]; // elitism
      const sigma = mut * (fn.hi - fn.lo);
      const blx = (u, v) => {
        const lo = Math.min(u, v),
          hi = Math.max(u, v),
          d = (hi - lo) * 0.5;
        return clamp(lo - d + rand() * (hi - lo + 2 * d), fn.lo, fn.hi);
      };
      let firstMate = null;
      while (next.length < n) {
        const a = tour(k),
          b = tour(k);
        for (let c = 0; c < 2 && next.length < n; c++) {
          let cx = blx(a.x, b.x),
            cy = blx(a.y, b.y);
          if (rand() < 0.5) cx = clamp(cx + gauss(rand) * sigma, fn.lo, fn.hi);
          if (rand() < 0.5) cy = clamp(cy + gauss(rand) * sigma, fn.lo, fn.hi);
          const child = { x: cx, y: cy, f: fn.f(cx, cy) };
          next.push(child);
          if (child.f < best.f) best = child;
          if (!firstMate) firstMate = { p1: a, p2: b, child };
        }
      }
      mating = firstMate;
      pop = next;
    },
    info() {
      return [
        ["best f", best.f.toExponential(2)],
        ["pop size", n],
        ["tournament k", this.params.k | 0],
        ["diversity", spread(pop, fn).toFixed(3)],
      ];
    },
    frame() {
      const sorted = [...pop].sort((a, b) => b.f - a.f);
      const dots = sorted.map((pt) => {
        const rank = 1 - sorted.indexOf(pt) / sorted.length;
        return {
          x: pt.x,
          y: pt.y,
          r: 3.6,
          color: `rgb(${Math.round(40 + rank * 70)},${Math.round(119 + rank * 30)},${Math.round(100 + rank * 40)})`,
        };
      });
      const rings = [
        { x: best.x, y: best.y, r: 7, color: COL.best, label: "elite" },
      ];
      const links = [];
      if (mating) {
        rings.push({
          x: mating.p1.x,
          y: mating.p1.y,
          r: 5,
          color: COL.warm,
          label: "parent",
        });
        rings.push({ x: mating.p2.x, y: mating.p2.y, r: 5, color: COL.warm });
        rings.push({
          x: mating.child.x,
          y: mating.child.y,
          r: 5,
          color: COL.cog,
          label: "child",
        });
        links.push({
          x1: mating.p1.x,
          y1: mating.p1.y,
          x2: mating.child.x,
          y2: mating.child.y,
          color: COL.cog,
          width: 1.7,
          dash: [4, 4],
          arrow: true,
        });
        links.push({
          x1: mating.p2.x,
          y1: mating.p2.y,
          x2: mating.child.x,
          y2: mating.child.y,
          color: COL.cog,
          width: 1.7,
          dash: [4, 4],
          arrow: true,
        });
      }
      return { dots, rings, links };
    },
  };
}

// ============================================================ Differential Evolution
export function createDE(fn, p) {
  const rand = rngFrom(p.seed ?? 2024);
  const n = p.n | 0,
    pop = [];
  for (let i = 0; i < n; i++) {
    const x = fn.lo + rand() * (fn.hi - fn.lo),
      y = fn.lo + rand() * (fn.hi - fn.lo);
    pop.push({ x, y, f: fn.f(x, y) });
  }
  let bi = 0;
  for (let i = 1; i < n; i++) if (pop[i].f < pop[bi].f) bi = i;
  let demo = null,
    meanDiff = 0;

  return {
    iter: 0,
    params: p,
    stepsPerTick: 1,
    get best() {
      return pop[bi].f;
    },
    status() {
      const s = spread(pop, fn);
      if (this.iter === 0)
        return "rand/1/bin. Watch the step size scale itself to the population.";
      if (s > 0.2)
        return "Wide population → large difference vectors → big exploratory moves.";
      if (s > 0.06)
        return "As the cloud contracts, |x_b − x_c| shrinks: steps self-scale down.";
      return "Difference vectors are small. The population may be refining or stagnating.";
    },
    step() {
      this.iter++;
      const { F, CR } = this.params;
      let diffSum = 0;
      for (let i = 0; i < n; i++) {
        let a, b, c;
        do {
          a = (rand() * n) | 0;
        } while (a === i);
        do {
          b = (rand() * n) | 0;
        } while (b === i || b === a);
        do {
          c = (rand() * n) | 0;
        } while (c === i || c === a || c === b);
        const donx = clamp(pop[a].x + F * (pop[b].x - pop[c].x), fn.lo, fn.hi);
        const dony = clamp(pop[a].y + F * (pop[b].y - pop[c].y), fn.lo, fn.hi);
        diffSum += Math.hypot(pop[b].x - pop[c].x, pop[b].y - pop[c].y);
        const forced = rand() < 0.5 ? 0 : 1;
        const tx = rand() < CR || forced === 0 ? donx : pop[i].x,
          ty = rand() < CR || forced === 1 ? dony : pop[i].y;
        const tf = fn.f(tx, ty);
        if (i === 0)
          demo = {
            a: pop[a],
            b: pop[b],
            c: pop[c],
            donor: { x: donx, y: dony },
            target: pop[i],
          };
        if (tf <= pop[i].f) {
          pop[i] = { x: tx, y: ty, f: tf };
          if (tf < pop[bi].f) bi = i;
        }
      }
      meanDiff = diffSum / n;
    },
    info() {
      return [
        ["best f", pop[bi].f.toExponential(2)],
        ["F", (+this.params.F).toFixed(2)],
        ["CR", (+this.params.CR).toFixed(2)],
        ["mean |x_b−x_c|", meanDiff.toFixed(3)],
      ];
    },
    frame() {
      const dots = pop.map((pt) => ({
        x: pt.x,
        y: pt.y,
        r: 3.4,
        color: COL.de,
      }));
      const rings = [
        { x: pop[bi].x, y: pop[bi].y, r: 7, color: COL.best, label: "best" },
      ];
      const links = [];
      if (demo) {
        rings.push({
          x: demo.a.x,
          y: demo.a.y,
          r: 5,
          color: COL.warm,
          label: "xₐ (base)",
        });
        rings.push({
          x: demo.b.x,
          y: demo.b.y,
          r: 4.5,
          color: COL.diff,
          label: "x_b",
        });
        rings.push({
          x: demo.c.x,
          y: demo.c.y,
          r: 4.5,
          color: COL.diff,
          label: "x_c",
        });
        rings.push({
          x: demo.donor.x,
          y: demo.donor.y,
          r: 6,
          color: COL.cog,
          label: "donor xₐ+F(x_b−x_c)",
        });
        links.push({
          x1: demo.c.x,
          y1: demo.c.y,
          x2: demo.b.x,
          y2: demo.b.y,
          color: COL.diff,
          width: 1.9,
          dash: [5, 4],
          arrow: true,
          label: "x_b − x_c",
        });
        links.push({
          x1: demo.a.x,
          y1: demo.a.y,
          x2: demo.donor.x,
          y2: demo.donor.y,
          color: COL.cog,
          width: 2.1,
          arrow: true,
          glow: true,
        });
      }
      return { dots, rings, links };
    },
  };
}

// ============================================================ Simulated Annealing
export function createSA(fn, p) {
  const rand = rngFrom(p.seed ?? 99);
  let x = fn.lo + rand() * (fn.hi - fn.lo),
    y = fn.lo + rand() * (fn.hi - fn.lo);
  let fx = fn.f(x, y),
    bx = x,
    by = y,
    bf = fx;
  let T = p.T0,
    lastD = 0,
    lastP = 1,
    lastAcc = true,
    proposal = null;
  const trail = [{ x, y, acc: true }];

  return {
    iter: 0,
    params: p,
    stepsPerTick: 1,
    trail,
    get best() {
      return bf;
    },
    get temp() {
      return T;
    },
    status() {
      if (this.iter === 0)
        return "Single walker. Worse moves accepted w.p. exp(−ΔE/T).";
      if (T > p.T0 * 0.15)
        return lastAcc && lastD > 0
          ? "Accepted a higher-value proposal. This can help cross a local barrier."
          : "Hot: exploring widely.";
      return "Cooled: ΔE>0 almost always rejected. Descending into a basin.";
    },
    step() {
      this.iter++;
      const { T0, cool } = this.params;
      const s = 0.16 * (fn.hi - fn.lo) * Math.sqrt(Math.max(T / T0, 1e-3));
      const nx = clamp(x + gauss(rand) * s, fn.lo, fn.hi),
        ny = clamp(y + gauss(rand) * s, fn.lo, fn.hi);
      const nf = fn.f(nx, ny),
        d = nf - fx;
      lastD = d;
      lastP = d < 0 ? 1 : Math.exp(-d / Math.max(T, 1e-9));
      lastAcc = d < 0 || rand() < lastP;
      proposal = { x, y, nx, ny };
      if (lastAcc) {
        x = nx;
        y = ny;
        fx = nf;
        if (fx < bf) {
          bf = fx;
          bx = x;
          by = y;
        }
      }
      trail.push({ x, y, acc: lastAcc });
      if (trail.length > 90) trail.shift();
      T *= cool;
      if (T < T0 * 1e-4) T = T0 * 1e-4;
    },
    info() {
      return [
        ["best f", bf.toExponential(2)],
        ["temperature T", T.toFixed(3)],
        ["last ΔE", (lastD >= 0 ? "+" : "") + lastD.toFixed(3)],
        [
          "P(accept) = e^(−ΔE/T)",
          lastD < 0 ? "1.00 (downhill)" : lastP.toFixed(3),
        ],
      ];
    },
    frame() {
      return {
        dots: [{ x, y, r: 5.5, color: COL.warm, glow: true }],
        rings: [
          { x: bx, y: by, r: 7, color: COL.best, label: "best" },
          { x, y, r: 0.1, color: COL.warm, label: "walker" },
        ],
        links: proposal
          ? [
              {
                x1: proposal.x,
                y1: proposal.y,
                x2: proposal.nx,
                y2: proposal.ny,
                color: lastAcc ? COL.cog : COL.reject,
                arrow: true,
                dash: lastAcc ? undefined : [4, 3],
              },
            ]
          : [],
        ...(proposal
          ? {
              rings: [
                {
                  x: proposal.x,
                  y: proposal.y,
                  r: 4,
                  color: COL.warm,
                  label: "current",
                },
                {
                  x: proposal.nx,
                  y: proposal.ny,
                  r: 5,
                  color: lastAcc ? COL.cog : COL.reject,
                  label: lastAcc ? "accepted" : "rejected",
                },
                { x: bx, y: by, r: 7, color: COL.best, label: "best" },
              ],
            }
          : {}),
      };
    },
  };
}

// ============================================================ CMA-ES
// A faithful 2-D (mu/mu_w, lambda) CMA-ES ; the same algorithm as
// metaheuristics/cma_es.py, specialised to n=2 so the sampling Gaussian can be
// drawn as an ellipse. Each generation: sample lambda points from N(m, sigma^2 C),
// keep the best mu, move the mean to their weighted average, then adapt sigma
// (step size) and C (shape) from the steps that worked.
export function createCMAES(fn, p) {
  const rand = rngFrom(p.seed ?? 4242);
  const n = 2;
  const lam = Math.max(4, p.n | 0);
  const mu = Math.max(1, lam >> 1);
  const wRaw = [];
  for (let i = 1; i <= mu; i++) wRaw.push(Math.log(mu + 0.5) - Math.log(i));
  const wSum = wRaw.reduce((s, v) => s + v, 0);
  const w = wRaw.map((v) => v / wSum);
  const muEff = 1 / w.reduce((s, v) => s + v * v, 0);
  const cS = (muEff + 2) / (n + muEff + 5);
  const dS = 1 + 2 * Math.max(0, Math.sqrt((muEff - 1) / (n + 1)) - 1) + cS;
  const cC = (4 + muEff / n) / (n + 4 + (2 * muEff) / n);
  const c1 = 2 / ((n + 1.3) ** 2 + muEff);
  const cMu = Math.min(
    1 - c1,
    (2 * (muEff - 2 + 1 / muEff)) / ((n + 2) ** 2 + muEff),
  );
  const chiN = Math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n));
  const span = fn.hi - fn.lo,
    box = (v) => clamp(v, fn.lo, fn.hi);

  let mx = fn.lo + (0.16 + 0.68 * rand()) * span,
    my = fn.lo + (0.16 + 0.68 * rand()) * span;
  let sigma = (p.sig ?? 0.3) * span;
  let c11 = 1,
    c12 = 0,
    c22 = 1;
  let pSx = 0,
    pSy = 0,
    pCx = 0,
    pCy = 0;
  let bx = mx,
    by = my,
    bf = fn.f(mx, my);
  let samples = [{ x: mx, y: my }],
    meanOld = { x: mx, y: my },
    axisRatio = 1;

  return {
    iter: 0,
    params: p,
    stepsPerTick: 1,
    get best() {
      return bf;
    },
    status() {
      if (this.iter === 0)
        return "Generation 0: an isotropic Gaussian N(m, σ²I) over the box.";
      if (axisRatio < 1.6)
        return "Sample λ, keep the best μ, slide the mean. The ellipse is still roughly round.";
      if (axisRatio < 4)
        return "The covariance is stretching along the direction that keeps paying off.";
      return "The sampling distribution is elongated: step sizes differ strongly by direction.";
    },
    step() {
      this.iter++;
      const E = eig2(c11, c12, c22);
      const D1 = Math.sqrt(E.l1),
        D2 = Math.sqrt(E.l2);
      axisRatio = Math.sqrt(Math.max(E.l1, E.l2) / Math.min(E.l1, E.l2));
      const cand = [];
      for (let k = 0; k < lam; k++) {
        const z1 = gauss(rand),
          z2 = gauss(rand);
        const yx = z1 * D1 * E.v1x + z2 * D2 * E.v2x;
        const yy = z1 * D1 * E.v1y + z2 * D2 * E.v2y; // y ~ N(0, C)
        const x = box(mx + sigma * yx),
          y = box(my + sigma * yy);
        cand.push({ x, y, yx, yy, f: fn.f(x, y) });
      }
      cand.sort((a, b) => a.f - b.f);
      if (cand[0].f < bf) {
        bf = cand[0].f;
        bx = cand[0].x;
        by = cand[0].y;
      }

      let ywx = 0,
        ywy = 0; // weighted recombination, in y-space
      for (let i = 0; i < mu; i++) {
        ywx += w[i] * cand[i].yx;
        ywy += w[i] * cand[i].yy;
      }
      meanOld = { x: mx, y: my };
      mx = box(mx + sigma * ywx);
      my = box(my + sigma * ywy);

      // step-size path: needs C^{-1/2} y_w, easy in the eigenbasis
      const a1 = ywx * E.v1x + ywy * E.v1y,
        a2 = ywx * E.v2x + ywy * E.v2y;
      const cinvx = (a1 / D1) * E.v1x + (a2 / D2) * E.v2x;
      const cinvy = (a1 / D1) * E.v1y + (a2 / D2) * E.v2y;
      const ks = Math.sqrt(cS * (2 - cS) * muEff);
      pSx = (1 - cS) * pSx + ks * cinvx;
      pSy = (1 - cS) * pSy + ks * cinvy;
      const pSn = Math.hypot(pSx, pSy);
      sigma *= Math.exp((cS / dS) * (pSn / chiN - 1));

      const hsig =
        pSn / Math.sqrt(1 - Math.pow(1 - cS, 2 * this.iter)) / chiN <
        1.4 + 2 / (n + 1)
          ? 1
          : 0;
      const kc = Math.sqrt(cC * (2 - cC) * muEff);
      pCx = (1 - cC) * pCx + hsig * kc * ywx;
      pCy = (1 - cC) * pCy + hsig * kc * ywy;

      let r11 = 0,
        r12 = 0,
        r22 = 0; // rank-mu
      for (let i = 0; i < mu; i++) {
        r11 += w[i] * cand[i].yx * cand[i].yx;
        r12 += w[i] * cand[i].yx * cand[i].yy;
        r22 += w[i] * cand[i].yy * cand[i].yy;
      }
      const dh = (1 - hsig) * cC * (2 - cC),
        keep = 1 - c1 - cMu;
      c11 = keep * c11 + c1 * (pCx * pCx + dh * c11) + cMu * r11;
      c12 = keep * c12 + c1 * (pCx * pCy + dh * c12) + cMu * r12;
      c22 = keep * c22 + c1 * (pCy * pCy + dh * c22) + cMu * r22;

      samples = cand.map((c) => ({ x: c.x, y: c.y }));
    },
    info() {
      return [
        ["best f", bf.toExponential(2)],
        ["σ (step)", sigma.toExponential(2)],
        ["axis ratio √(λ₁/λ₂)", axisRatio.toFixed(2)],
        ["λ / μ", `${lam} / ${mu}`],
      ];
    },
    frame() {
      const E = eig2(c11, c12, c22);
      const D1 = Math.sqrt(E.l1),
        D2 = Math.sqrt(E.l2);
      const Ax = sigma * D1 * E.v1x,
        Ay = sigma * D1 * E.v1y; // 1σ semi-axes, domain coords
      const Bx = sigma * D2 * E.v2x,
        By = sigma * D2 * E.v2y;
      const dots = samples.map((s) => ({
        x: s.x,
        y: s.y,
        r: 3,
        color: COL.cma,
      }));
      const ellipses = [
        {
          cx: mx,
          cy: my,
          ax: 2 * Ax,
          ay: 2 * Ay,
          bx: 2 * Bx,
          by: 2 * By,
          color: COL.cma,
          alpha: 0.3,
        },
        {
          cx: mx,
          cy: my,
          ax: Ax,
          ay: Ay,
          bx: Bx,
          by: By,
          color: COL.cma,
          alpha: 0.85,
          fill: true,
        },
      ];
      const rings = [
        { x: mx, y: my, r: 5, color: COL.cma, label: "mean m" },
        { x: bx, y: by, r: 7, color: COL.best, label: "best" },
      ];
      const links =
        this.iter > 0
          ? [
              {
                x1: meanOld.x,
                y1: meanOld.y,
                x2: mx,
                y2: my,
                color: COL.best,
                width: 1.6,
                dash: [4, 3],
                arrow: true,
              },
            ]
          : [];
      return { dots, rings, links, ellipses };
    },
  };
}
