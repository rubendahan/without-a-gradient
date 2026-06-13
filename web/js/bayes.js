// 1-D Bayesian optimization demo: a hidden expensive function, a Gaussian-process
// surrogate (mean + uncertainty band), the Expected-Improvement acquisition, and
// the next point it picks. Step through it one expensive evaluation at a time.
import { rngFrom, gauss } from "./landscape.js";

const erf = (x) => {
  // Abramowitz & Stegun 7.1.26
  const t = 1 / (1 + 0.3275911 * Math.abs(x));
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return x >= 0 ? y : -y;
};
const ncdf = (z) => 0.5 * (1 + erf(z / Math.SQRT2));
const npdf = (z) => Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);

// The hidden objective (we pretend it's expensive). Domain [0, 1].
function trueF(x) {
  return -(Math.sin(3 * Math.PI * x) * 0.6 + Math.exp(-Math.pow((x - 0.78) / 0.12, 2)) * 0.9
    - Math.pow(x - 0.5, 2) * 0.8);
}

export function createBayes(p) {
  const rand = rngFrom(p.seed ?? 7);
  const lo = 0, hi = 1;
  const noise = 1e-4;
  const X = [], Y = [];
  // seed with 2 random points
  for (let i = 0; i < 2; i++) { const x = lo + rand() * (hi - lo); X.push(x); Y.push(trueF(x)); }

  const GRID = 240;
  const grid = Array.from({ length: GRID }, (_, i) => lo + (i / (GRID - 1)) * (hi - lo));

  let post = null, nextX = null, ei = null;

  function kernel(a, b, l, sf) {
    const d = a - b; return sf * Math.exp(-0.5 * d * d / (l * l));
  }
  // tiny Cholesky solve
  function chol(A) {
    const n = A.length, L = Array.from({ length: n }, () => new Float64Array(n));
    for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
      let s = A[i][j]; for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      L[i][j] = i === j ? Math.sqrt(Math.max(s, 1e-12)) : s / L[j][j];
    }
    return L;
  }
  function solveL(L, b) { const n = L.length, y = new Float64Array(n); for (let i = 0; i < n; i++) { let s = b[i]; for (let k = 0; k < i; k++) s -= L[i][k] * y[k]; y[i] = s / L[i][i]; } return y; }
  function solveLT(L, b) { const n = L.length, x = new Float64Array(n); for (let i = n - 1; i >= 0; i--) { let s = b[i]; for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k]; x[i] = s / L[i][i]; } return x; }

  function refit() {
    const l = p.length, sf = 1.0, n = X.length;
    const ymean = Y.reduce((a, b) => a + b, 0) / n;
    const yc = Y.map((v) => v - ymean);
    const K = Array.from({ length: n }, (_, i) => new Float64Array(n));
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) K[i][j] = kernel(X[i], X[j], l, sf) + (i === j ? noise : 0);
    const L = chol(K);
    const alpha = solveLT(L, solveL(L, yc));
    const mean = new Float64Array(GRID), sd = new Float64Array(GRID);
    for (let g = 0; g < GRID; g++) {
      const ks = new Float64Array(n);
      for (let i = 0; i < n; i++) ks[i] = kernel(X[i], grid[g], l, sf);
      let m = 0; for (let i = 0; i < n; i++) m += ks[i] * alpha[i];
      mean[g] = m + ymean;
      const v = solveL(L, ks);
      let var_ = sf; for (let i = 0; i < n; i++) var_ -= v[i] * v[i];
      sd[g] = Math.sqrt(Math.max(var_, 1e-9));
    }
    post = { mean, sd };
    // Expected improvement (minimization), xi exploration margin
    const fbest = Math.min(...Y), xi = p.xi;
    ei = new Float64Array(GRID); let bestEI = -1, bi = 0;
    for (let g = 0; g < GRID; g++) {
      const s = sd[g], mu = mean[g];
      if (s < 1e-9) { ei[g] = 0; continue; }
      const z = (fbest - xi - mu) / s;
      ei[g] = (fbest - xi - mu) * ncdf(z) + s * npdf(z);
      if (ei[g] > bestEI) { bestEI = ei[g]; bi = g; }
    }
    nextX = grid[bi];
  }
  refit();

  return {
    iter: 0,
    get best() { return Math.min(...Y); },
    params: p,
    onParam() { refit(); },
    step() {
      if (nextX == null) return;
      this.iter++;
      X.push(nextX); Y.push(trueF(nextX));
      refit();
    },
    draw(ctx, W, H) {
      ctx.clearRect(0, 0, W, H);
      const padL = 8, padR = 8, padT = 12, padB = 60;
      const plotH = H - padT - padB, plotW = W - padL - padR;
      // y-range from true function + data
      let ymin = Infinity, ymax = -Infinity;
      for (let g = 0; g < GRID; g++) { const t = trueF(grid[g]); ymin = Math.min(ymin, t, post.mean[g] - 2 * post.sd[g]); ymax = Math.max(ymax, t, post.mean[g] + 2 * post.sd[g]); }
      const yr = ymax - ymin || 1;
      const X2px = (x) => padL + ((x - lo) / (hi - lo)) * plotW;
      const Y2px = (y) => padT + (1 - (y - ymin) / yr) * plotH;

      // uncertainty band (mean +/- 2 sd)
      ctx.fillStyle = "rgba(88,166,255,0.18)";
      ctx.beginPath();
      for (let g = 0; g < GRID; g++) { const x = X2px(grid[g]), y = Y2px(post.mean[g] + 2 * post.sd[g]); g === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); }
      for (let g = GRID - 1; g >= 0; g--) { ctx.lineTo(X2px(grid[g]), Y2px(post.mean[g] - 2 * post.sd[g])); }
      ctx.closePath(); ctx.fill();

      // true function (dashed)
      ctx.strokeStyle = "rgba(154,166,178,0.55)"; ctx.lineWidth = 1.5; ctx.setLineDash([5, 4]);
      ctx.beginPath();
      for (let g = 0; g < GRID; g++) { const x = X2px(grid[g]), y = Y2px(trueF(grid[g])); g === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); }
      ctx.stroke(); ctx.setLineDash([]);

      // GP mean
      ctx.strokeStyle = "#58a6ff"; ctx.lineWidth = 2;
      ctx.beginPath();
      for (let g = 0; g < GRID; g++) { const x = X2px(grid[g]), y = Y2px(post.mean[g]); g === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); }
      ctx.stroke();

      // observations
      for (let i = 0; i < X.length; i++) {
        ctx.fillStyle = "#f0b72f";
        ctx.beginPath(); ctx.arc(X2px(X[i]), Y2px(Y[i]), 4, 0, Math.PI * 2); ctx.fill();
      }

      // EI strip at the bottom
      const eiTop = H - padB + 14, eiH = padB - 22;
      let eimax = Math.max(...ei) || 1;
      ctx.fillStyle = "rgba(86,211,100,0.25)";
      ctx.beginPath(); ctx.moveTo(X2px(grid[0]), eiTop + eiH);
      for (let g = 0; g < GRID; g++) ctx.lineTo(X2px(grid[g]), eiTop + eiH - (ei[g] / eimax) * eiH);
      ctx.lineTo(X2px(grid[GRID - 1]), eiTop + eiH); ctx.closePath(); ctx.fill();
      ctx.strokeStyle = "#56d364"; ctx.lineWidth = 1.5;
      ctx.beginPath();
      for (let g = 0; g < GRID; g++) { const x = X2px(grid[g]), y = eiTop + eiH - (ei[g] / eimax) * eiH; g === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); }
      ctx.stroke();

      // next-point marker
      if (nextX != null) {
        const nx = X2px(nextX);
        ctx.strokeStyle = "rgba(255,255,255,0.6)"; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.moveTo(nx, padT); ctx.lineTo(nx, eiTop + eiH); ctx.stroke(); ctx.setLineDash([]);
      }

      ctx.fillStyle = "#6b7684"; ctx.font = "11px monospace";
      ctx.fillText("dashed = hidden f   blue = GP mean   band = ±2σ", padL, H - 30);
      ctx.fillStyle = "#56d364";
      ctx.fillText("green = Expected Improvement → next sample (white line)", padL, H - 14);
    },
  };
}
