// Builds the interactive widgets. Core idea: the algorithm advances in discrete
// steps, but the harness TWEENS between successive frame()s with easing, so the
// motion is slow and readable. A live state panel surfaces the maths.
import {
  FUNCS,
  makeMapper,
  renderLandscape,
  setupCanvas,
  tag,
  lerp,
  easeInOut,
} from "./landscape.js";
import {
  createPSO,
  createGA,
  createDE,
  createSA,
  createCMAES,
} from "./algos.js";
import { createBayes } from "./bayes.js";
import { mountTrap } from "./intro.js";
import { renderLoss, renderOperation } from "./figures.js";

const SPEEDS = { slow: 1400, normal: 900, fast: 480 };

function Sld(id, label, min, max, step, value, hint, fmt) {
  return { id, label, min, max, step, value, hint, fmt: fmt || ((v) => v) };
}

const CONFIGS = {
  pso: {
    title: "particle swarm",
    factory: createPSO,
    trails: true,
    funcKeys: ["rastrigin", "ackley", "himmelblau", "rosenbrock"],
    defaults: { n: 22, w: 0.72, c1: 1.5, c2: 1.5 },
    controls: [
      Sld("n", "swarm size", 6, 60, 1, 22, "number of particles", (v) => v | 0),
      Sld(
        "w",
        "inertia  w",
        0,
        1.2,
        0.01,
        0.72,
        "momentum: high explores, low exploits",
        (v) => v.toFixed(2),
      ),
      Sld(
        "c1",
        "cognitive  c₁",
        0,
        3,
        0.05,
        1.5,
        "attraction to own best pᵢ",
        (v) => v.toFixed(2),
      ),
      Sld(
        "c2",
        "social  c₂",
        0,
        3,
        0.05,
        1.5,
        "attraction to global best ĝ",
        (v) => v.toFixed(2),
      ),
    ],
    legend: [
      ["#2864a5", "particle xᵢ"],
      ["#658bb4", "velocity vᵢ"],
      ["#278362", "pull to pᵢ"],
      ["#2864a5", "pull to ĝ"],
      ["#223447", "global best ĝ"],
    ],
  },
  ga: {
    title: "genetic algorithm",
    factory: createGA,
    funcKeys: ["rastrigin", "ackley", "himmelblau", "sphere"],
    defaults: { n: 36, mut: 0.08, k: 3 },
    controls: [
      Sld(
        "n",
        "population",
        12,
        90,
        2,
        36,
        "individuals per generation",
        (v) => v | 0,
      ),
      Sld(
        "mut",
        "mutation σ",
        0.005,
        0.3,
        0.005,
        0.08,
        "Gaussian perturbation scale",
        (v) => v.toFixed(3),
      ),
      Sld("k", "tournament k", 2, 8, 1, 3, "selection pressure", (v) => v | 0),
    ],
    legend: [
      ["#278362", "fitter"],
      ["#6e8c3a", "weaker"],
      ["#ae7024", "parents"],
      ["#223447", "elite"],
    ],
  },
  de: {
    title: "differential evolution",
    factory: createDE,
    trails: true,
    funcKeys: ["rastrigin", "ackley", "rosenbrock", "himmelblau"],
    defaults: { n: 28, F: 0.7, CR: 0.9 },
    controls: [
      Sld(
        "n",
        "population",
        8,
        70,
        1,
        28,
        "members in the cloud",
        (v) => v | 0,
      ),
      Sld(
        "F",
        "weight  F",
        0.1,
        1.2,
        0.05,
        0.7,
        "scale of the difference vector",
        (v) => v.toFixed(2),
      ),
      Sld(
        "CR",
        "crossover  CR",
        0.1,
        1,
        0.05,
        0.9,
        "per-coordinate mixing rate",
        (v) => v.toFixed(2),
      ),
    ],
    legend: [
      ["#79529a", "candidate"],
      ["#8760a2", "x_b, x_c"],
      ["#278362", "donor"],
      ["#223447", "best"],
    ],
  },
  sa: {
    title: "simulated annealing",
    factory: createSA,
    funcKeys: ["rastrigin", "ackley", "himmelblau", "sphere"],
    defaults: { T0: 4, cool: 0.985 },
    controls: [
      Sld(
        "T0",
        "initial temp  T₀",
        0.2,
        12,
        0.1,
        4,
        "higher = more uphill moves early",
        (v) => v.toFixed(1),
      ),
      Sld(
        "cool",
        "cooling  α",
        0.9,
        0.999,
        0.001,
        0.985,
        "T ← αT each step",
        (v) => v.toFixed(3),
      ),
    ],
    legend: [
      ["#ae7024", "walker / accepted"],
      ["#c35938", "rejected"],
      ["#223447", "best"],
    ],
  },
  cmaes: {
    title: "CMA-ES",
    factory: createCMAES,
    funcKeys: ["rosenbrock", "ackley", "himmelblau", "sphere"],
    defaults: { n: 12, sig: 0.3 },
    rebuildOn: ["n", "sig"],
    controls: [
      Sld(
        "n",
        "samples  λ",
        6,
        40,
        1,
        12,
        "candidates drawn per generation",
        (v) => v | 0,
      ),
      Sld(
        "sig",
        "initial step  σ₀",
        0.08,
        0.6,
        0.02,
        0.3,
        "as a fraction of the box width",
        (v) => v.toFixed(2),
      ),
    ],
    legend: [
      ["#287f85", "sample xₖ"],
      ["#287f85", "1σ / 2σ ellipse"],
      ["#223447", "mean m / best"],
    ],
  },
};

// ---- generic frame tween + draw ----------------------------------------
function tweenFrame(a, b, e) {
  const arr = (pa, pb, mix) => pb.map((q, i) => (pa[i] ? mix(pa[i], q) : q));
  return {
    dots: arr(a.dots, b.dots, (p, q) => ({
      ...q,
      x: lerp(p.x, q.x, e),
      y: lerp(p.y, q.y, e),
    })),
    rings: arr(a.rings || [], b.rings || [], (p, q) => ({
      ...q,
      x: lerp(p.x, q.x, e),
      y: lerp(p.y, q.y, e),
    })),
    links: arr(a.links || [], b.links || [], (p, q) => ({
      ...q,
      x1: lerp(p.x1, q.x1, e),
      y1: lerp(p.y1, q.y1, e),
      x2: lerp(p.x2, q.x2, e),
      y2: lerp(p.y2, q.y2, e),
    })),
    ellipses: arr(a.ellipses || [], b.ellipses || [], (p, q) => ({
      ...q,
      cx: lerp(p.cx, q.cx, e),
      cy: lerp(p.cy, q.cy, e),
      ax: lerp(p.ax, q.ax, e),
      ay: lerp(p.ay, q.ay, e),
      bx: lerp(p.bx, q.bx, e),
      by: lerp(p.by, q.by, e),
    })),
  };
}

const TAU = Math.PI * 2;

function arrowHead(ctx, x1, y1, x2, y2, color, size = 7) {
  const a = Math.atan2(y2 - y1, x2 - x1);
  ctx.save();
  ctx.fillStyle = color;
  ctx.shadowBlur = 0;
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - size * Math.cos(a - 0.42), y2 - size * Math.sin(a - 0.42));
  ctx.lineTo(x2 - size * Math.cos(a + 0.42), y2 - size * Math.sin(a + 0.42));
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

// Fading comet trails: a list of past resolved frames + the live one, connected
// per dot-index. Only valid where dot index == a stable agent (PSO, DE).
function drawTrails(ctx, map, frames) {
  if (frames.length < 2) return;
  const last = frames[frames.length - 1].dots;
  ctx.save();
  ctx.lineCap = "round";
  for (let k = 0; k < last.length; k++) {
    for (let i = 1; i < frames.length; i++) {
      const a = frames[i - 1].dots[k],
        b = frames[i].dots[k];
      if (!a || !b) continue;
      const [x1, y1] = map.toPx(a.x, a.y),
        [x2, y2] = map.toPx(b.x, b.y);
      ctx.strokeStyle = b.color || "#2864a5";
      ctx.globalAlpha = 0.05 + 0.2 * (i / (frames.length - 1));
      ctx.lineWidth = 0.6 + 1.6 * (i / (frames.length - 1));
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }
  }
  ctx.restore();
}

function drawFrame(ctx, map, fr, pulse = 0) {
  // sampling ellipses (CMA-ES): trace the contour in domain coords and map each
  // point, so the curve is correct even when px/unit differs on the two axes.
  for (const el of fr.ellipses || []) {
    ctx.save();
    ctx.beginPath();
    const N = 60;
    for (let i = 0; i <= N; i++) {
      const t = (i / N) * TAU;
      const dx = el.cx + Math.cos(t) * el.ax + Math.sin(t) * el.bx;
      const dy = el.cy + Math.cos(t) * el.ay + Math.sin(t) * el.by;
      const [px, py] = map.toPx(dx, dy);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.closePath();
    if (el.fill) {
      ctx.globalAlpha = 0.1;
      ctx.fillStyle = el.color;
      ctx.fill();
    }
    ctx.globalAlpha = el.alpha != null ? el.alpha : 0.8;
    ctx.strokeStyle = el.color;
    ctx.lineWidth = 1.6;
    if (el.fill) {
      ctx.shadowColor = el.color;
      ctx.shadowBlur = 0;
    }
    ctx.stroke();
    ctx.restore();
  }
  for (const l of fr.links || []) {
    const [x1, y1] = map.toPx(l.x1, l.y1),
      [x2, y2] = map.toPx(l.x2, l.y2);
    ctx.save();
    ctx.strokeStyle = l.color;
    ctx.lineWidth = l.width || 1.6;
    ctx.globalAlpha = l.alpha != null ? l.alpha : 0.92;
    ctx.lineCap = "round";
    if (l.dash) ctx.setLineDash(l.dash);
    if (l.glow) {
      ctx.shadowColor = l.color;
      ctx.shadowBlur = 0;
    }
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    ctx.setLineDash([]);
    if (l.arrow && Math.hypot(x2 - x1, y2 - y1) > 6)
      arrowHead(ctx, x1, y1, x2, y2, l.color);
    ctx.restore();
  }
  for (const d of fr.dots) {
    const [x, y] = map.toPx(d.x, d.y);
    ctx.save();
    // soft halo
    ctx.globalAlpha = 0;
    ctx.fillStyle = d.color;
    ctx.beginPath();
    ctx.arc(x, y, d.r * 2.6, 0, TAU);
    ctx.fill();
    // glowing core
    ctx.globalAlpha = 1;
    if (d.glow) {
      ctx.shadowColor = d.color;
      ctx.shadowBlur = 0;
    }
    ctx.fillStyle = d.color;
    ctx.beginPath();
    ctx.arc(x, y, d.r, 0, TAU);
    ctx.fill();
    // specular highlight
    ctx.shadowBlur = 0;
    ctx.fillStyle = "rgba(255,255,255,0)";
    ctx.beginPath();
    ctx.arc(x - d.r * 0.28, y - d.r * 0.28, d.r * 0.38, 0, TAU);
    ctx.fill();
    ctx.restore();
  }
  for (const r of fr.rings || []) {
    const [x, y] = map.toPx(r.x, r.y);
    if (r.r > 1) {
      ctx.save();
      if (r.color === "#223447") {
        // pulsing halo on the incumbent best
        ctx.globalAlpha = 0.22 + 0.22 * pulse;
        ctx.strokeStyle = "#223447";
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        ctx.arc(x, y, r.r + 4 + 4 * pulse, 0, TAU);
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
      ctx.shadowColor = "rgba(0,0,0,0.5)";
      ctx.shadowBlur = 5;
      ctx.strokeStyle = r.color;
      ctx.lineWidth = 2.4;
      ctx.beginPath();
      ctx.arc(x, y, r.r, 0, TAU);
      ctx.stroke();
      if (r.color === "#223447") {
        ctx.fillStyle = "#223447";
        ctx.beginPath();
        ctx.arc(x, y, 2, 0, TAU);
        ctx.fill();
      }
      ctx.restore();
    }
  }
  // Geometry is identified in the legend, avoiding overlapping labels.
}

function buildControls(cfg) {
  const wrap = document.createElement("div");
  wrap.className = "controls";
  const fnCtrl = document.createElement("div");
  fnCtrl.className = "ctrl";
  fnCtrl.innerHTML = `<label><span>objective</span></label>`;
  const sel = document.createElement("select");
  sel.setAttribute("aria-label", `${cfg.title}: objective`);
  cfg.funcKeys.forEach((k) => {
    const o = document.createElement("option");
    o.value = k;
    o.textContent = `${FUNCS[k].label} · ${FUNCS[k].blurb}`;
    sel.appendChild(o);
  });
  fnCtrl.appendChild(sel);
  wrap.appendChild(fnCtrl);

  const advanced = document.createElement("details");
  advanced.className = "parameter-settings";
  advanced.innerHTML = "<summary>Algorithm parameters</summary>";
  const inputs = {};
  for (const c of cfg.controls) {
    const ctrl = document.createElement("div");
    ctrl.className = "ctrl";
    const lab = document.createElement("label");
    const val = document.createElement("span");
    val.className = "val";
    lab.innerHTML = `<span>${c.label}</span>`;
    lab.appendChild(val);
    const inp = document.createElement("input");
    inp.setAttribute("aria-label", `${cfg.title}: ${c.label}`);
    inp.type = "range";
    inp.min = c.min;
    inp.max = c.max;
    inp.step = c.step;
    inp.value = c.value;
    const hint = document.createElement("div");
    hint.className = "hint";
    hint.textContent = c.hint;
    const sync = () => {
      val.textContent = c.fmt(parseFloat(inp.value));
    };
    inp.addEventListener("input", sync);
    sync();
    ctrl.append(lab, inp, hint);
    advanced.appendChild(ctrl);
    inputs[c.id] = inp;
  }

  wrap.appendChild(advanced);
  const spd = document.createElement("div");
  spd.className = "ctrl";
  spd.innerHTML = `<label><span>speed</span></label>`;
  const spdSel = document.createElement("select");
  spdSel.setAttribute("aria-label", `${cfg.title}: animation speed`);
  ["slow", "normal", "fast"].forEach((s) => {
    const o = document.createElement("option");
    o.value = s;
    o.textContent = s;
    if (s === "normal") o.selected = true;
    spdSel.appendChild(o);
  });
  spd.appendChild(spdSel);
  wrap.appendChild(spd);

  const btns = document.createElement("div");
  btns.className = "btns";
  const play = document.createElement("button");
  play.className = "btn";
  play.textContent = "Run";
  const step = document.createElement("button");
  step.className = "btn ghost";
  step.textContent = "Step";
  const reset = document.createElement("button");
  reset.className = "btn ghost";
  reset.textContent = "Reset";
  btns.append(play, step, reset);
  wrap.appendChild(btns);

  const panel = document.createElement("details");
  panel.className = "statepanel";
  panel.innerHTML = `<summary class="sp-title">Numerical state</summary><div class="sp-body"></div>`;
  wrap.appendChild(panel);

  const legend = document.createElement("div");
  legend.className = "legend";
  for (const [color, text] of cfg.legend) {
    const s = document.createElement("span");
    s.innerHTML = `<span class="dot" style="background:${color}"></span>${text}`;
    legend.appendChild(s);
  }
  wrap.appendChild(legend);
  return {
    wrap,
    sel,
    inputs,
    spdSel,
    play,
    step,
    reset,
    spBody: panel.querySelector(".sp-body"),
  };
}

function mount2D(root, key) {
  const cfg = CONFIGS[key];
  root.classList.add("simulation");
  root.innerHTML = `<div class="demo-head"><span class="title">${cfg.title}</span><span class="stat"></span></div>`;
  const stat = root.querySelector(".stat"),
    C = buildControls(cfg);
  const toolbar = document.createElement("div");
  toolbar.className = "simulation-toolbar";
  toolbar.append(C.sel.parentElement, C.play.parentElement);
  const grid = document.createElement("div");
  grid.className = "simulation-grid";
  grid.innerHTML = `<div class="population-view"><div class="canvas-wrap"><canvas role="img" aria-label="${cfg.title} population on an objective landscape"></canvas></div>
    <div class="map-key"><span class="map-scale">Lower f <i style="background:#b8cedf"></i><i style="background:#dce7ef"></i><i style="background:#f7f9fb"></i> Higher f</span><span>◇ Known optimum</span></div></div>
    <div class="analysis-view"><div class="loss-heading"><div><span>Best loss</span><strong class="loss-value"></strong></div><span class="loss-change"></span></div>
    <div class="loss-figure"><svg viewBox="0 0 360 185" role="img" aria-label="Best loss against objective evaluations"></svg><div class="loss-axis-note"><span>Logarithmic scale: log₁₀(1 + f)</span><span>Evaluations →</span></div></div>
    <div class="operation-figure"><div class="operation-caption"></div><svg viewBox="0 0 360 160" role="img"></svg><span class="operation-scale">Current run · detail scaled independently</span></div></div>`;
  const canvas = grid.querySelector("canvas"),
    chart = grid.querySelector(".loss-figure svg");
  const operation = grid.querySelector(".operation-figure svg"),
    opCaption = grid.querySelector(".operation-caption");
  const lossValue = grid.querySelector(".loss-value"),
    lossChange = grid.querySelector(".loss-change");
  const narr = document.createElement("div");
  narr.className = "narration";
  const advanced = C.wrap.querySelector(".parameter-settings");
  advanced.querySelector("summary").textContent =
    "Parameters & reproducibility";
  const seedCtrl = document.createElement("div");
  seedCtrl.className = "ctrl";
  seedCtrl.innerHTML = `<label for="seed-${key}">Random seed</label><select id="seed-${key}"><option value="12345">12345</option><option value="42">42</option><option value="2026">2026</option></select><div class="hint">Reset replays the same seed.</div>`;
  advanced.append(seedCtrl, C.spdSel.parentElement);
  const seedSelect = seedCtrl.querySelector("select");
  const footer = document.createElement("div");
  footer.className = "simulation-footer";
  footer.append(C.wrap.querySelector(".legend"), advanced);
  // Detailed values belong with optional parameters, not in the default view.
  advanced.append(C.spBody.parentElement);
  root.append(toolbar, grid, narr, footer);
  let running = false,
    animating = false,
    inView = false,
    algo,
    fn,
    map,
    bg,
    ctx,
    W,
    H,
    prev,
    curr;
  let history = [],
    scores = [],
    evaluations = 0,
    tStart = 0,
    tickMs = SPEEDS.normal,
    ticks = 0,
    duration = SPEEDS.normal;
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const valueFormat = (v) =>
    Math.abs(v) >= 0.001 && Math.abs(v) < 10000
      ? Number(v.toPrecision(5)).toString()
      : v.toExponential(3);
  function setRunning(value) {
    running = value;
    C.play.textContent = value ? "Pause" : "Play";
    C.play.setAttribute("aria-pressed", String(value));
    C.play.disabled = ticks >= 200;
    C.step.disabled = ticks >= 200;
  }
  function resize() {
    const d = setupCanvas(canvas);
    ctx = d.ctx;
    W = d.W;
    H = d.H;
    bg = renderLandscape(fn, W, H);
    map = makeMapper(fn, W, H);
    paint(1);
  }
  function rebuild() {
    evaluations = 0;
    ticks = 0;
    animating = false;
    setRunning(false);
    const base = FUNCS[C.sel.value];
    const objective = {
      ...base,
      f: (x, y) => {
        evaluations++;
        return base.f(x, y);
      },
    };
    const params = { ...cfg.defaults, seed: Number(seedSelect.value) };
    for (const id in C.inputs) params[id] = Number(C.inputs[id].value);
    algo = cfg.factory(objective, params);
    fn = base;
    prev = algo.frame();
    curr = prev;
    history = [];
    scores = [{ n: evaluations, f: algo.best }];
    resize();
    refresh();
    paint(1);
  }
  function paint(e) {
    if (!ctx || !curr) return;
    ctx.clearRect(0, 0, W, H);
    ctx.drawImage(bg, 0, 0, W, H);
    ctx.save();
    ctx.beginPath();
    ctx.rect(map.left, map.top, map.size, map.size);
    ctx.clip();
    if (algo.trail) {
      for (let i = 1; i < algo.trail.length; i++) {
        const a = algo.trail[i - 1],
          b = algo.trail[i];
        const [x1, y1] = map.toPx(a.x, a.y),
          [x2, y2] = map.toPx(b.x, b.y);
        ctx.strokeStyle = b.acc ? "#ad762b66" : "#ba583b55";
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
      }
    }
    if (["ga", "cmaes"].includes(key)) {
      // New samples fade in, rather than pretending offspring travel from parents.
      if (e < 1) {
        ctx.globalAlpha = (1 - e) * 0.35;
        drawFrame(ctx, map, { dots: prev.dots }, 0);
      }
      ctx.globalAlpha = 0.3 + 0.7 * e;
      drawFrame(ctx, map, { ...tweenFrame(prev, curr, e), dots: curr.dots }, 0);
    } else {
      const fr = tweenFrame(prev, curr, e);
      if (cfg.trails) drawTrails(ctx, map, [...history, fr]);
      drawFrame(ctx, map, fr, 0);
    }
    ctx.restore();
    renderLoss(chart, scores, e);
    renderOperation(operation, opCaption, key, prev, curr, algo.iter, e);
  }
  function refresh() {
    stat.textContent = `Iteration ${algo.iter} · ${evaluations} evaluations`;
    lossValue.textContent = valueFormat(algo.best);
    const initial = scores[0].f;
    const reduction =
      initial > 1e-12 ? (100 * (initial - algo.best)) / initial : 0;
    lossChange.textContent = ticks
      ? `${reduction.toFixed(1)}% below start`
      : "Lower is better";
    C.spBody.innerHTML = algo
      .info()
      .map(([k, v]) => `<div class="sp-row"><span>${k}</span><b>${v}</b></div>`)
      .join("");
    narr.textContent = algo.status();
    chart.setAttribute(
      "aria-label",
      `Best loss ${algo.best.toPrecision(4)} after ${evaluations} evaluations. Logarithmic vertical scale.`,
    );
    canvas.setAttribute(
      "aria-label",
      `${cfg.title} on ${fn.label}. Iteration ${algo.iter}, best loss ${algo.best.toPrecision(4)} after ${evaluations} evaluations.`,
    );
  }
  function advance(single = false) {
    if (ticks >= 200) {
      setRunning(false);
      return;
    }
    ticks++;
    algo.step();
    history.push(curr);
    if (history.length > 16) history.shift();
    prev = curr;
    curr = algo.frame();
    scores.push({ n: evaluations, f: algo.best });
    tStart = performance.now();
    duration = single ? 1300 : tickMs;
    animating = !reduced;
    refresh();
    paint(reduced ? 1 : 0);
    if (ticks >= 200) {
      setRunning(false);
      narr.textContent =
        "200 iterations complete. The final population and full loss curve remain visible. Reset to replay.";
    }
  }
  C.sel.addEventListener("change", rebuild);
  seedSelect.addEventListener("change", rebuild);
  for (const input of Object.values(C.inputs))
    input.addEventListener("input", rebuild);
  C.spdSel.addEventListener("change", () => {
    tickMs = SPEEDS[C.spdSel.value];
  });
  C.play.addEventListener("click", () => {
    setRunning(!running);
    if (running && !animating) advance();
  });
  C.step.addEventListener("click", () => {
    setRunning(false);
    advance(true);
  });
  C.reset.addEventListener("click", rebuild);
  new IntersectionObserver(
    (es) => {
      inView = es[0].isIntersecting;
      if (inView) {
        tStart = performance.now();
      }
    },
    { threshold: 0.05 },
  ).observe(root);
  let rt;
  new ResizeObserver(() => {
    clearTimeout(rt);
    rt = setTimeout(resize, 100);
  }).observe(canvas.parentElement);
  function loop(now) {
    if (inView && !document.hidden) {
      if (animating) {
        const t = Math.min(1, (now - tStart) / duration);
        paint(easeInOut(t));
        if (t >= 1) {
          animating = false;
          if (running) advance();
        }
      } else if (running && now - tStart >= duration) advance();
    }
    requestAnimationFrame(loop);
  }
  rebuild();
  requestAnimationFrame(loop);
}

// ---------------- Bayesian (1-D, step-driven) ----------------
function mountBayes(root) {
  root.classList.add("simulation", "bayesian-simulation");
  root.innerHTML = `<div class="demo-head"><span class="title">Bayesian optimisation</span><span class="stat"></span></div>
    <div class="simulation-toolbar"><span class="bayes-objective">Gaussian process · expected improvement</span><div class="btns"><button class="btn auto">Play</button><button class="btn ghost sample">Step</button><button class="btn ghost reset">Reset</button></div></div>
    <div class="simulation-grid"><div class="population-view"><div class="canvas-wrap"><canvas role="img" aria-label="Gaussian process posterior and expected improvement"></canvas></div></div>
    <div class="analysis-view"><div class="loss-heading"><div><span>Best loss</span><strong class="loss-value"></strong></div><span class="loss-change">Lower is better</span></div><div class="loss-figure"><svg viewBox="0 0 360 185" role="img" aria-label="Best observed loss against evaluations"></svg><div class="loss-axis-note"><span>Linear scale</span><span>Evaluations →</span></div></div>
    <div class="bayes-explanation"><strong>One evaluation</strong><p><i>1</i> Fit the model to the observed values.</p><p><i>2</i> Find the largest expected improvement.</p><p><i>3</i> Evaluate that point and update the model.</p><span>The vertical line marks the next query. The lower green curve explains why it was chosen.</span></div></div></div>
    <div class="narration"></div><div class="simulation-footer"><div class="legend"><span><i class="dot" style="background:#637487"></i>Dashed: objective</span><span><i class="dot" style="background:#2864a5"></i>Model mean ± 2σ</span><span><i class="dot" style="background:#ae7024"></i>Observations</span><span><i class="dot" style="background:#278362"></i>Expected improvement</span></div>
    <details class="parameter-settings"><summary>Model parameters</summary><div class="ctrl"><label for="bayes-length">Length scale <span id="bls">0.12</span></label><input id="bayes-length" type="range" min="0.03" max="0.4" step="0.01" value="0.12"><div class="hint">How far an observation influences the model.</div></div><div class="ctrl"><label for="bayes-xi">Exploration margin <span id="bxi">0.010</span></label><input id="bayes-xi" type="range" min="0" max="0.2" step="0.005" value="0.01"><div class="hint">A larger margin favours uncertain regions.</div></div><p class="hint">Fixed seed 7; finite acquisition grid; 32-observation limit. Reset replays the same start.</p></details></div>`;
  const canvas = root.querySelector("canvas"),
    stat = root.querySelector(".stat"),
    narr = root.querySelector(".narration");
  const length = root.querySelector("#bayes-length"),
    xi = root.querySelector("#bayes-xi");
  const sample = root.querySelector(".sample"),
    autoBtn = root.querySelector(".auto"),
    chart = root.querySelector(".loss-figure svg");
  let algo,
    ctx,
    W,
    H,
    auto = false,
    inView = false,
    last = 0,
    scores = [];
  function setAuto(value) {
    auto = value;
    autoBtn.textContent = value ? "Pause" : "Play";
    autoBtn.setAttribute("aria-pressed", String(value));
    last = performance.now();
  }
  function render() {
    algo.draw(ctx, W, H);
    stat.textContent = `${algo.iter + 2} evaluations`;
    root.querySelector(".loss-value").textContent = algo.best.toFixed(4);
    renderLoss(chart, scores, 1, true);
    chart.setAttribute(
      "aria-label",
      `Best observed loss ${algo.best.toFixed(4)} after ${algo.iter + 2} evaluations. Linear scale.`,
    );
    canvas.setAttribute(
      "aria-label",
      `Bayesian model after ${algo.iter + 2} observations. Best loss ${algo.best.toFixed(4)}.`,
    );
  }
  function resize() {
    const d = setupCanvas(canvas);
    ctx = d.ctx;
    W = d.W;
    H = d.H;
    if (algo) render();
  }
  function rebuild() {
    setAuto(false);
    algo = createBayes({ seed: 7, length: +length.value, xi: +xi.value });
    scores = [{ n: 2, f: algo.best }];
    sample.disabled = false;
    autoBtn.disabled = false;
    narr.textContent =
      "Two initial observations. Play runs the search; Step adds exactly one evaluation.";
    resize();
  }
  function step() {
    if (algo.iter >= 30) return;
    algo.step();
    scores.push({ n: algo.iter + 2, f: algo.best });
    render();
    narr.textContent = `Observed ${algo.iter + 2} points. The vertical dashed line marks the next proposed query.`;
    if (algo.iter >= 30) {
      setAuto(false);
      sample.disabled = true;
      autoBtn.disabled = true;
      narr.textContent =
        "32 observations reached. The model and loss curve remain visible. Reset to replay.";
    }
  }
  length.addEventListener("input", () => {
    root.querySelector("#bls").textContent = (+length.value).toFixed(2);
    algo.params.length = +length.value;
    algo.onParam();
    render();
  });
  xi.addEventListener("input", () => {
    root.querySelector("#bxi").textContent = (+xi.value).toFixed(3);
    algo.params.xi = +xi.value;
    algo.onParam();
    render();
  });
  sample.addEventListener("click", () => {
    setAuto(false);
    step();
  });
  autoBtn.addEventListener("click", () => {
    setAuto(!auto);
    if (auto) step();
  });
  root.querySelector(".reset").addEventListener("click", rebuild);
  new IntersectionObserver((es) => {
    inView = es[0].isIntersecting;
    last = performance.now();
  }).observe(root);
  new ResizeObserver(resize).observe(canvas.parentElement);
  function loop(now) {
    if (auto && inView && !document.hidden && now - last >= 1100) {
      step();
      last = now;
    }
    requestAnimationFrame(loop);
  }
  rebuild();
  requestAnimationFrame(loop);
}

function drawCover() {
  const canvas = document.querySelector("#cover-canvas");
  if (!canvas) return;
  const d = setupCanvas(canvas),
    fn = FUNCS.rosenbrock,
    map = makeMapper(fn, d.W, d.H);
  d.ctx.drawImage(renderLandscape(fn, d.W, d.H), 0, 0);
  const algo = createPSO(fn, { n: 18, w: 0.72, c1: 1.5, c2: 1.5, seed: 42 });
  const frames = [algo.frame()];
  for (let i = 0; i < 24; i++) {
    algo.step();
    frames.push(algo.frame());
  }
  drawTrails(d.ctx, map, frames);
  drawFrame(
    d.ctx,
    map,
    { dots: frames.at(-1).dots, rings: frames.at(-1).rings },
    0,
  );
}

function boot() {
  drawCover();
  new ResizeObserver(drawCover).observe(
    document.querySelector("#cover-canvas").parentElement,
  );
  document.querySelectorAll(".demo[data-demo]").forEach((el) => {
    const key = el.getAttribute("data-demo");
    if (key === "trap") mountTrap(el);
    else if (key === "bayes") mountBayes(el);
    else if (CONFIGS[key]) mount2D(el, key);
  });
  const links = [...document.querySelectorAll("nav.toc a")];
  const map = new Map(links.map((a) => [a.getAttribute("href").slice(1), a]));
  const obs = new IntersectionObserver(
    (es) =>
      es.forEach((e) => {
        if (e.isIntersecting) {
          links.forEach((a) => a.classList.remove("active"));
          const a = map.get(e.target.id);
          if (a) {
            a.classList.add("active");
            links.forEach((link) => link.removeAttribute("aria-current"));
            a.setAttribute("aria-current", "location");
          }
        }
      }),
    { rootMargin: "-40% 0px -55% 0px" },
  );
  document.querySelectorAll("section[id]").forEach((s) => obs.observe(s));
}

if (document.readyState === "loading")
  document.addEventListener("DOMContentLoaded", boot);
else boot();
