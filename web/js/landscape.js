// Shared utilities for every demo: 2-D test functions, a contour/heatmap
// renderer, and pixel <-> domain coordinate mapping. No dependencies.

export const TWO_PI = Math.PI * 2;

// --- 2-D benchmark landscapes (minimization). Each carries its own box. ---
export const FUNCS = {
  rastrigin: {
    label: "Rastrigin",
    lo: -5.12, hi: 5.12,
    opt: [0, 0],
    f: (x, y) =>
      20 + (x * x - 10 * Math.cos(TWO_PI * x)) + (y * y - 10 * Math.cos(TWO_PI * y)),
    blurb: "a grid of ~100 local minima",
  },
  ackley: {
    label: "Ackley",
    lo: -5, hi: 5,
    opt: [0, 0],
    f: (x, y) => {
      const s1 = x * x + y * y;
      const s2 = Math.cos(TWO_PI * x) + Math.cos(TWO_PI * y);
      return -20 * Math.exp(-0.2 * Math.sqrt(s1 / 2)) - Math.exp(s2 / 2) + 20 + Math.E;
    },
    blurb: "flat plateau, one deep central funnel",
  },
  himmelblau: {
    label: "Himmelblau",
    lo: -5, hi: 5,
    opt: [3, 2],
    f: (x, y) => Math.pow(x * x + y - 11, 2) + Math.pow(x + y * y - 7, 2),
    blurb: "four equal global minima",
  },
  rosenbrock: {
    label: "Rosenbrock",
    lo: -2, hi: 2,
    opt: [1, 1],
    f: (x, y) => 100 * Math.pow(y - x * x, 2) + Math.pow(1 - x, 2),
    blurb: "a narrow curved valley",
  },
  sphere: {
    label: "Sphere",
    lo: -5, hi: 5,
    opt: [0, 0],
    f: (x, y) => x * x + y * y,
    blurb: "a single convex bowl",
  },
};

// Map a domain point to canvas pixels and back, given a function's box.
export function makeMapper(fn, W, H) {
  const { lo, hi } = fn;
  const span = hi - lo;
  return {
    toPx: (x, y) => [((x - lo) / span) * W, H - ((y - lo) / span) * H],
    toDom: (px, py) => [lo + (px / W) * span, lo + ((H - py) / H) * span],
    lo, hi, span,
  };
}

// A perceptual blue->green->yellow->red ramp for f-values in [0,1].
function ramp(t) {
  t = Math.max(0, Math.min(1, t));
  const stops = [
    [13, 27, 52], [22, 64, 120], [33, 145, 140],
    [94, 201, 98], [241, 196, 47], [220, 90, 60],
  ];
  const s = t * (stops.length - 1);
  const i = Math.floor(s);
  const f = s - i;
  const a = stops[i], b = stops[Math.min(i + 1, stops.length - 1)];
  return [
    Math.round(a[0] + (b[0] - a[0]) * f),
    Math.round(a[1] + (b[1] - a[1]) * f),
    Math.round(a[2] + (b[2] - a[2]) * f),
  ];
}

// Precompute a heatmap (with faint contour lines) as an ImageData-backed canvas.
// Returns an offscreen canvas you can drawImage() each frame -- cheap.
export function renderLandscape(fn, W, H) {
  const off = document.createElement("canvas");
  off.width = W; off.height = H;
  const ctx = off.getContext("2d");
  const img = ctx.createImageData(W, H);
  const { lo, hi } = fn;
  const span = hi - lo;

  // First pass: sample to find value range for normalization (log-compressed).
  let vmin = Infinity, vmax = -Infinity;
  const STEP = 2;
  const vals = new Float32Array((W / STEP) * (H / STEP) | 0);
  let k = 0;
  for (let py = 0; py < H; py += STEP) {
    for (let px = 0; px < W; px += STEP) {
      const x = lo + (px / W) * span;
      const y = lo + ((H - py) / H) * span;
      const v = Math.log1p(Math.max(0, fn.f(x, y)));
      vals[k++] = v;
      if (v < vmin) vmin = v;
      if (v > vmax) vmax = v;
    }
  }
  const rng = vmax - vmin || 1;

  for (let py = 0; py < H; py++) {
    for (let px = 0; px < W; px++) {
      const x = lo + (px / W) * span;
      const y = lo + ((H - py) / H) * span;
      const v = Math.log1p(Math.max(0, fn.f(x, y)));
      const t = (v - vmin) / rng;
      let [r, g, b] = ramp(t);
      // Faint contour banding for depth perception.
      const band = (t * 14) % 1;
      if (band < 0.06) { r *= 0.78; g *= 0.78; b *= 0.78; }
      const idx = (py * W + px) * 4;
      img.data[idx] = r; img.data[idx + 1] = g; img.data[idx + 2] = b; img.data[idx + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);

  // Mark the global optimum with a small ring.
  const map = makeMapper(fn, W, H);
  const [ox, oy] = map.toPx(fn.opt[0], fn.opt[1]);
  ctx.strokeStyle = "rgba(255,255,255,0.55)";
  ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.arc(ox, oy, 7, 0, TWO_PI); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(ox - 11, oy); ctx.lineTo(ox - 4, oy);
  ctx.moveTo(ox + 4, oy); ctx.lineTo(ox + 11, oy);
  ctx.moveTo(ox, oy - 11); ctx.lineTo(ox, oy - 4);
  ctx.moveTo(ox, oy + 4); ctx.lineTo(ox, oy + 11); ctx.stroke();

  return off;
}

// Make a canvas crisp on HiDPI screens; returns {ctx, W, H} in CSS pixels.
export function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const W = Math.max(320, Math.floor(rect.width));
  const H = Math.max(340, Math.floor(rect.height || 360));
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, W, H };
}

// Seedable RNG (mulberry32) so demos are reproducible per-run.
export function rngFrom(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function gauss(rand) {
  // Box-Muller.
  let u = 0, v = 0;
  while (u === 0) u = rand();
  while (v === 0) v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(TWO_PI * v);
}

// Wire a range input to a live value label + callback.
export function bindSlider(input, label, fmt, onChange) {
  const update = () => {
    const v = parseFloat(input.value);
    if (label) label.textContent = fmt ? fmt(v) : v;
    if (onChange) onChange(v);
  };
  input.addEventListener("input", update);
  update();
  return () => parseFloat(input.value);
}
