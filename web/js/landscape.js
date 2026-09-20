// Shared rendering toys: the 2-D objective landscapes, a clean topographic
// "depth map" renderer (dark basins = low/good, contour lines for legibility),
// pixel <-> domain mapping, easing, and small drawing helpers. No dependencies.

export const TWO_PI = Math.PI * 2;
export const lerp = (a, b, t) => a + (b - a) * t;
export const easeInOut = (t) =>
  t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;

// f: R^2 -> R, minimized. Low = good. Each carries its own box and optimum.
export const FUNCS = {
  rastrigin: {
    label: "Rastrigin",
    lo: -5.12,
    hi: 5.12,
    opt: [0, 0],
    f: (x, y) =>
      20 +
      (x * x - 10 * Math.cos(TWO_PI * x)) +
      (y * y - 10 * Math.cos(TWO_PI * y)),
    blurb: "many local minima",
  },
  ackley: {
    label: "Ackley",
    lo: -5,
    hi: 5,
    opt: [0, 0],
    f: (x, y) => {
      const s1 = x * x + y * y,
        s2 = Math.cos(TWO_PI * x) + Math.cos(TWO_PI * y);
      return (
        -20 * Math.exp(-0.2 * Math.sqrt(s1 / 2)) -
        Math.exp(s2 / 2) +
        20 +
        Math.E
      );
    },
    blurb: "flat outer plateau, central funnel",
  },
  himmelblau: {
    label: "Himmelblau",
    lo: -5,
    hi: 5,
    opt: [3, 2],
    f: (x, y) => Math.pow(x * x + y - 11, 2) + Math.pow(x + y * y - 7, 2),
    blurb: "four equal global optima",
  },
  rosenbrock: {
    label: "Rosenbrock",
    lo: -2,
    hi: 2,
    opt: [1, 1],
    f: (x, y) => 100 * Math.pow(y - x * x, 2) + Math.pow(1 - x, 2),
    blurb: "ill-conditioned curved valley",
  },
  sphere: {
    label: "Sphere",
    lo: -5,
    hi: 5,
    opt: [0, 0],
    f: (x, y) => x * x + y * y,
    blurb: "convex, unimodal",
  },
};

export function makeMapper(fn, W, H) {
  const span = fn.hi - fn.lo;
  const size = Math.max(20, Math.min(W - 82, H - 65));
  const left = Math.max(42, (W - size) / 2 - 5),
    top = 22;
  return {
    toPx: (x, y) => [
      left + ((x - fn.lo) / span) * size,
      top + size - ((y - fn.lo) / span) * size,
    ],
    toDom: (x, y) => [
      fn.lo + ((x - left) / size) * span,
      fn.lo + ((top + size - y) / size) * span,
    ],
    lo: fn.lo,
    hi: fn.hi,
    span,
    left,
    top,
    size,
  };
}

// Equal-scale axes, a pale sequential fill and explicit logarithmic contours.
export function renderLandscape(fn, W, H) {
  const off = document.createElement("canvas");
  off.width = W;
  off.height = H;
  const ctx = off.getContext("2d"),
    map = makeMapper(fn, W, H);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, W, H);
  const N = 120,
    values = new Float64Array((N + 1) * (N + 1));
  let max = 0;
  for (let j = 0; j <= N; j++)
    for (let i = 0; i <= N; i++) {
      const v = Math.log10(
        1 +
          Math.max(
            0,
            fn.f(fn.lo + (i / N) * map.span, fn.hi - (j / N) * map.span),
          ),
      );
      values[j * (N + 1) + i] = v;
      max = Math.max(max, v);
    }
  const cell = map.size / N;
  for (let j = 0; j < N; j++)
    for (let i = 0; i < N; i++) {
      const t = Math.pow(values[j * (N + 1) + i] / (max || 1), 0.65);
      ctx.fillStyle = `rgb(${Math.round(184 + 65 * t)},${Math.round(206 + 44 * t)},${Math.round(224 + 28 * t)})`;
      ctx.fillRect(
        map.left + i * cell,
        map.top + j * cell,
        cell + 0.4,
        cell + 0.4,
      );
    }
  // Marching squares, using a consistent cell diagonal for ambiguous cases.
  ctx.lineWidth = 0.75;
  ctx.strokeStyle = "#6887a788";
  for (let level = 1; level < 15; level++) {
    const threshold = (max * level) / 15;
    ctx.beginPath();
    for (let j = 0; j < N; j++)
      for (let i = 0; i < N; i++) {
        const corners = [
          [i, j],
          [i + 1, j],
          [i + 1, j + 1],
          [i, j + 1],
        ];
        const vs = corners.map(([x, y]) => values[y * (N + 1) + x]);
        const hits = [];
        for (let k = 0; k < 4; k++) {
          const l = (k + 1) % 4;
          if (vs[k] < threshold !== vs[l] < threshold) {
            const t = (threshold - vs[k]) / (vs[l] - vs[k]);
            hits.push([
              map.left +
                (corners[k][0] + t * (corners[l][0] - corners[k][0])) * cell,
              map.top +
                (corners[k][1] + t * (corners[l][1] - corners[k][1])) * cell,
            ]);
          }
        }
        for (let k = 0; k + 1 < hits.length; k += 2) {
          ctx.moveTo(...hits[k]);
          ctx.lineTo(...hits[k + 1]);
        }
      }
    ctx.stroke();
  }
  ctx.strokeStyle = "#9eafbf";
  ctx.lineWidth = 0.8;
  ctx.strokeRect(map.left, map.top, map.size, map.size);
  ctx.fillStyle = "#58687a";
  ctx.font = "10px Segoe UI, sans-serif";
  for (let i = 0; i <= 4; i++) {
    const value = fn.lo + (i / 4) * map.span,
      x = map.left + (i / 4) * map.size,
      y = map.top + map.size - (i / 4) * map.size;
    const label = Number(value.toFixed(2)).toString();
    ctx.textAlign = "center";
    ctx.fillText(label, x, map.top + map.size + 18);
    ctx.textAlign = "right";
    ctx.fillText(label, map.left - 9, y + 3);
  }
  ctx.textAlign = "center";
  ctx.font = "italic 13px Georgia, serif";
  ctx.fillText("x₁", map.left + map.size / 2, map.top + map.size + 37);
  ctx.fillText("x₂", map.left - 28, map.top + map.size / 2);
  ctx.textAlign = "left";
  ctx.font = "10px Segoe UI, sans-serif";
  ctx.fillText(fn.label, map.left, 12);
  ctx.textAlign = "right";
  ctx.fillText("Contours: log₁₀(1 + f)", map.left + map.size, 12);
  const opts =
    fn.label === "Himmelblau"
      ? [
          [3, 2],
          [-2.805118, 3.131312],
          [-3.77931, -3.283186],
          [3.584428, -1.848126],
        ]
      : [fn.opt];
  for (const point of opts) {
    const [x, y] = map.toPx(...point);
    ctx.fillStyle = "#fff";
    ctx.strokeStyle = "#263e55";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x, y - 5);
    ctx.lineTo(x + 5, y);
    ctx.lineTo(x, y + 5);
    ctx.lineTo(x - 5, y);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  }
  return off;
}

// pill label
export function tag(ctx, text, x, y, bg, fg) {
  ctx.save();
  ctx.font = "600 11px ui-monospace, monospace";
  const w = ctx.measureText(text).width + 12;
  const cw = ctx.canvas.width / (ctx.getTransform().a || 1);
  x = Math.max(4, Math.min(x, cw - w - 4));
  if (y < 12) y += 24;
  ctx.fillStyle = bg;
  roundRect(ctx, x, y - 11, w, 17, 5);
  ctx.fill();
  ctx.fillStyle = fg;
  ctx.textBaseline = "middle";
  ctx.fillText(text, x + 6, y - 2);
  ctx.restore();
}

export function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

export function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const W = Math.max(1, Math.floor(rect.width));
  const H = Math.max(1, Math.floor(rect.height || 400));
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, W, H };
}

export function rngFrom(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
export function gauss(rand) {
  let u = 0,
    v = 0;
  while (u === 0) u = rand();
  while (v === 0) v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(TWO_PI * v);
}
