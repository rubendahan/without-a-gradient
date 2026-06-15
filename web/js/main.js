// Builds the interactive widgets. Core idea: the algorithm advances in discrete
// steps, but the harness TWEENS between successive frame()s with easing, so the
// motion is slow and readable. A live state panel surfaces the maths.
import { FUNCS, makeMapper, renderLandscape, setupCanvas, tag, lerp, easeInOut } from "./landscape.js";
import { createPSO, createGA, createDE, createSA, createCMAES } from "./algos.js";
import { createBayes } from "./bayes.js";
import { mountTrap } from "./intro.js";

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
      Sld("w", "inertia  w", 0, 1.2, 0.01, 0.72, "momentum: high explores, low exploits", (v) => v.toFixed(2)),
      Sld("c1", "cognitive  c₁", 0, 3, 0.05, 1.5, "attraction to own best pᵢ", (v) => v.toFixed(2)),
      Sld("c2", "social  c₂", 0, 3, 0.05, 1.5, "attraction to global best ĝ", (v) => v.toFixed(2)),
    ],
    legend: [["#5ec8ff", "particle xᵢ"], ["#9fd8ff", "velocity vᵢ"], ["#56d364", "pull to pᵢ"], ["#5ec8ff", "pull to ĝ"], ["#fff", "global best ĝ"]],
  },
  ga: {
    title: "genetic algorithm",
    factory: createGA,
    funcKeys: ["rastrigin", "ackley", "himmelblau", "sphere"],
    defaults: { n: 36, mut: 0.08, k: 3 },
    controls: [
      Sld("n", "population", 12, 90, 2, 36, "individuals per generation", (v) => v | 0),
      Sld("mut", "mutation σ", 0.005, 0.3, 0.005, 0.08, "Gaussian perturbation scale", (v) => v.toFixed(3)),
      Sld("k", "tournament k", 2, 8, 1, 3, "selection pressure", (v) => v | 0),
    ],
    legend: [["#56d364", "fitter"], ["#b0e06e", "weaker"], ["#f0b72f", "parents"], ["#fff", "elite"]],
  },
  de: {
    title: "differential evolution",
    factory: createDE,
    trails: true,
    funcKeys: ["rastrigin", "ackley", "rosenbrock", "himmelblau"],
    defaults: { n: 28, F: 0.7, CR: 0.9 },
    controls: [
      Sld("n", "population", 8, 70, 1, 28, "members in the cloud", (v) => v | 0),
      Sld("F", "weight  F", 0.1, 1.2, 0.05, 0.7, "scale of the difference vector", (v) => v.toFixed(2)),
      Sld("CR", "crossover  CR", 0.1, 1, 0.05, 0.9, "per-coordinate mixing rate", (v) => v.toFixed(2)),
    ],
    legend: [["#c8a6ff", "candidate"], ["#d2a8ff", "x_b, x_c"], ["#56d364", "donor"], ["#fff", "best"]],
  },
  sa: {
    title: "simulated annealing",
    factory: createSA,
    funcKeys: ["rastrigin", "ackley", "himmelblau", "sphere"],
    defaults: { T0: 4, cool: 0.985 },
    controls: [
      Sld("T0", "initial temp  T₀", 0.2, 12, 0.1, 4, "higher = more uphill moves early", (v) => v.toFixed(1)),
      Sld("cool", "cooling  α", 0.9, 0.999, 0.001, 0.985, "T ← αT each step", (v) => v.toFixed(3)),
    ],
    legend: [["#f0b72f", "walker / accepted"], ["#ff7b72", "rejected"], ["#fff", "best"]],
  },
  cmaes: {
    title: "CMA-ES",
    factory: createCMAES,
    funcKeys: ["rosenbrock", "ackley", "himmelblau", "sphere"],
    defaults: { n: 12, sig: 0.3 },
    rebuildOn: ["n", "sig"],
    controls: [
      Sld("n", "samples  λ", 6, 40, 1, 12, "candidates drawn per generation", (v) => v | 0),
      Sld("sig", "initial step  σ₀", 0.08, 0.6, 0.02, 0.3, "as a fraction of the box width", (v) => v.toFixed(2)),
    ],
    legend: [["#3ad0c0", "sample xₖ"], ["#3ad0c0", "1σ / 2σ ellipse"], ["#fff", "mean m / best"]],
  },
};

// ---- generic frame tween + draw ----------------------------------------
function tweenFrame(a, b, e) {
  const arr = (pa, pb, mix) => pb.map((q, i) => (pa[i] ? mix(pa[i], q) : q));
  return {
    dots: arr(a.dots, b.dots, (p, q) => ({ ...q, x: lerp(p.x, q.x, e), y: lerp(p.y, q.y, e) })),
    rings: arr(a.rings || [], b.rings || [], (p, q) => ({ ...q, x: lerp(p.x, q.x, e), y: lerp(p.y, q.y, e) })),
    links: arr(a.links || [], b.links || [], (p, q) => ({
      ...q, x1: lerp(p.x1, q.x1, e), y1: lerp(p.y1, q.y1, e), x2: lerp(p.x2, q.x2, e), y2: lerp(p.y2, q.y2, e),
    })),
    ellipses: arr(a.ellipses || [], b.ellipses || [], (p, q) => ({
      ...q, cx: lerp(p.cx, q.cx, e), cy: lerp(p.cy, q.cy, e),
      ax: lerp(p.ax, q.ax, e), ay: lerp(p.ay, q.ay, e), bx: lerp(p.bx, q.bx, e), by: lerp(p.by, q.by, e),
    })),
  };
}

const TAU = Math.PI * 2;

function arrowHead(ctx, x1, y1, x2, y2, color, size = 7) {
  const a = Math.atan2(y2 - y1, x2 - x1);
  ctx.save();
  ctx.fillStyle = color; ctx.shadowBlur = 0;
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - size * Math.cos(a - 0.42), y2 - size * Math.sin(a - 0.42));
  ctx.lineTo(x2 - size * Math.cos(a + 0.42), y2 - size * Math.sin(a + 0.42));
  ctx.closePath(); ctx.fill();
  ctx.restore();
}

// Fading comet trails: a list of past resolved frames + the live one, connected
// per dot-index. Only valid where dot index == a stable agent (PSO, DE).
function drawTrails(ctx, map, frames) {
  if (frames.length < 2) return;
  const last = frames[frames.length - 1].dots;
  ctx.save(); ctx.lineCap = "round";
  for (let k = 0; k < last.length; k++) {
    for (let i = 1; i < frames.length; i++) {
      const a = frames[i - 1].dots[k], b = frames[i].dots[k];
      if (!a || !b) continue;
      const [x1, y1] = map.toPx(a.x, a.y), [x2, y2] = map.toPx(b.x, b.y);
      ctx.strokeStyle = b.color || "#5ec8ff";
      ctx.globalAlpha = 0.05 + 0.2 * (i / (frames.length - 1));
      ctx.lineWidth = 0.6 + 1.6 * (i / (frames.length - 1));
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
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
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.closePath();
    if (el.fill) { ctx.globalAlpha = 0.1; ctx.fillStyle = el.color; ctx.fill(); }
    ctx.globalAlpha = el.alpha != null ? el.alpha : 0.8;
    ctx.strokeStyle = el.color; ctx.lineWidth = 1.6;
    if (el.fill) { ctx.shadowColor = el.color; ctx.shadowBlur = 8; }
    ctx.stroke();
    ctx.restore();
  }
  for (const l of fr.links || []) {
    const [x1, y1] = map.toPx(l.x1, l.y1), [x2, y2] = map.toPx(l.x2, l.y2);
    ctx.save();
    ctx.strokeStyle = l.color; ctx.lineWidth = l.width || 1.6;
    ctx.globalAlpha = l.alpha != null ? l.alpha : 0.92;
    ctx.lineCap = "round";
    if (l.dash) ctx.setLineDash(l.dash);
    if (l.glow) { ctx.shadowColor = l.color; ctx.shadowBlur = 9; }
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    ctx.setLineDash([]);
    if (l.arrow && Math.hypot(x2 - x1, y2 - y1) > 6) arrowHead(ctx, x1, y1, x2, y2, l.color);
    ctx.restore();
  }
  for (const d of fr.dots) {
    const [x, y] = map.toPx(d.x, d.y);
    ctx.save();
    // soft halo
    ctx.globalAlpha = 0.16; ctx.fillStyle = d.color;
    ctx.beginPath(); ctx.arc(x, y, d.r * 2.6, 0, TAU); ctx.fill();
    // glowing core
    ctx.globalAlpha = 1;
    if (d.glow) { ctx.shadowColor = d.color; ctx.shadowBlur = 14; }
    ctx.fillStyle = d.color;
    ctx.beginPath(); ctx.arc(x, y, d.r, 0, TAU); ctx.fill();
    // specular highlight
    ctx.shadowBlur = 0; ctx.fillStyle = "rgba(255,255,255,0.5)";
    ctx.beginPath(); ctx.arc(x - d.r * 0.28, y - d.r * 0.28, d.r * 0.38, 0, TAU); ctx.fill();
    ctx.restore();
  }
  for (const r of fr.rings || []) {
    const [x, y] = map.toPx(r.x, r.y);
    if (r.r > 1) {
      ctx.save();
      if (r.color === "#ffffff") { // pulsing halo on the incumbent best
        ctx.globalAlpha = 0.22 + 0.22 * pulse; ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.4;
        ctx.beginPath(); ctx.arc(x, y, r.r + 4 + 4 * pulse, 0, TAU); ctx.stroke();
        ctx.globalAlpha = 1;
      }
      ctx.shadowColor = "rgba(0,0,0,0.5)"; ctx.shadowBlur = 5;
      ctx.strokeStyle = r.color; ctx.lineWidth = 2.4;
      ctx.beginPath(); ctx.arc(x, y, r.r, 0, TAU); ctx.stroke();
      if (r.color === "#ffffff") { ctx.fillStyle = "#fff"; ctx.beginPath(); ctx.arc(x, y, 2, 0, TAU); ctx.fill(); }
      ctx.restore();
    }
  }
  // labels last, on top
  for (const r of fr.rings || []) if (r.label) { const [x, y] = map.toPx(r.x, r.y); tag(ctx, r.label, x + r.r + 5, y - r.r - 3, "rgba(12,18,30,0.82)", r.color === "#ffffff" ? "#fff" : r.color); }
  for (const l of fr.links || []) if (l.label) { const mx = (l.x1 + l.x2) / 2, my = (l.y1 + l.y2) / 2; const [x, y] = map.toPx(mx, my); tag(ctx, l.label, x + 4, y, "rgba(12,18,30,0.82)", l.color); }
}

function buildControls(cfg) {
  const wrap = document.createElement("div"); wrap.className = "controls";
  const fnCtrl = document.createElement("div"); fnCtrl.className = "ctrl";
  fnCtrl.innerHTML = `<label><span>objective</span></label>`;
  const sel = document.createElement("select");
  cfg.funcKeys.forEach((k) => { const o = document.createElement("option"); o.value = k; o.textContent = `${FUNCS[k].label} · ${FUNCS[k].blurb}`; sel.appendChild(o); });
  fnCtrl.appendChild(sel); wrap.appendChild(fnCtrl);

  const inputs = {};
  for (const c of cfg.controls) {
    const ctrl = document.createElement("div"); ctrl.className = "ctrl";
    const lab = document.createElement("label");
    const val = document.createElement("span"); val.className = "val";
    lab.innerHTML = `<span>${c.label}</span>`; lab.appendChild(val);
    const inp = document.createElement("input");
    inp.type = "range"; inp.min = c.min; inp.max = c.max; inp.step = c.step; inp.value = c.value;
    const hint = document.createElement("div"); hint.className = "hint"; hint.textContent = c.hint;
    const sync = () => { val.textContent = c.fmt(parseFloat(inp.value)); };
    inp.addEventListener("input", sync); sync();
    ctrl.append(lab, inp, hint); wrap.appendChild(ctrl); inputs[c.id] = inp;
  }

  const spd = document.createElement("div"); spd.className = "ctrl";
  spd.innerHTML = `<label><span>speed</span></label>`;
  const spdSel = document.createElement("select");
  ["slow", "normal", "fast"].forEach((s) => { const o = document.createElement("option"); o.value = s; o.textContent = s; if (s === "normal") o.selected = true; spdSel.appendChild(o); });
  spd.appendChild(spdSel); wrap.appendChild(spd);

  const btns = document.createElement("div"); btns.className = "btns";
  const play = document.createElement("button"); play.className = "btn"; play.textContent = "❚❚";
  const step = document.createElement("button"); step.className = "btn ghost"; step.textContent = "step ⏭";
  const reset = document.createElement("button"); reset.className = "btn ghost"; reset.textContent = "↻";
  btns.append(play, step, reset); wrap.appendChild(btns);

  const panel = document.createElement("div"); panel.className = "statepanel";
  panel.innerHTML = `<div class="sp-title">live state</div><div class="sp-body"></div>`;
  wrap.appendChild(panel);

  const legend = document.createElement("div"); legend.className = "legend";
  for (const [color, text] of cfg.legend) { const s = document.createElement("span"); s.innerHTML = `<span class="dot" style="background:${color}"></span>${text}`; legend.appendChild(s); }
  wrap.appendChild(legend);
  return { wrap, sel, inputs, spdSel, play, step, reset, spBody: panel.querySelector(".sp-body") };
}

function mount2D(root, key) {
  const cfg = CONFIGS[key];
  root.innerHTML = "";
  const head = document.createElement("div"); head.className = "demo-head";
  head.innerHTML = `<span class="title">${cfg.title}</span><span class="stat">·</span>`;
  const stat = head.querySelector(".stat");
  const grid = document.createElement("div"); grid.className = "demo-grid";
  const col = document.createElement("div"); col.className = "canvas-col";
  const cw = document.createElement("div"); cw.className = "canvas-wrap";
  const canvas = document.createElement("canvas"); cw.appendChild(canvas);
  const narr = document.createElement("div"); narr.className = "narration"; narr.textContent = "…";
  col.append(cw, narr);
  const C = buildControls(cfg);
  grid.append(col, C.wrap);
  root.append(head, grid);

  const TRAIL = 8;
  let running = true, algo, fn, map, bg, ctx, W, H, prev, curr, history = [], tStart = 0, tickMs = SPEEDS.normal;
  // The demo animates only while it is on screen, and after it has converged it
  // pauses for a beat and restarts, so it is always running from the start when
  // you actually look at it.
  let inView = false, started = false, finishing = false, restartAt = 0, ticks = 0;
  const LOOP_TICKS = 55;
  const readParams = () => { const p = { ...cfg.defaults }; for (const id in C.inputs) p[id] = parseFloat(C.inputs[id].value); return p; };

  function rebuild() {
    fn = FUNCS[C.sel.value];
    const d = setupCanvas(canvas); ctx = d.ctx; W = d.W; H = d.H;
    bg = renderLandscape(fn, W, H); map = makeMapper(fn, W, H);
    algo = cfg.factory(fn, readParams());
    prev = algo.frame(); curr = prev; history = []; tStart = performance.now();
    ticks = 0; finishing = false;
    paint(0);
  }
  function paint(e) {
    ctx.clearRect(0, 0, W, H); ctx.drawImage(bg, 0, 0, W, H);
    if (algo.trail) {                                   // SA history (untweened)
      ctx.lineWidth = 1.4;
      for (let i = 1; i < algo.trail.length; i++) {
        const a = algo.trail[i - 1], b = algo.trail[i];
        const [a0, b0] = map.toPx(a.x, a.y), [a1, b1] = map.toPx(b.x, b.y);
        const al = i / algo.trail.length;
        ctx.strokeStyle = b.acc ? `rgba(240,183,47,${0.15 + 0.6 * al})` : `rgba(255,123,114,${0.2 + 0.3 * al})`;
        ctx.beginPath(); ctx.moveTo(a0, b0); ctx.lineTo(a1, b1); ctx.stroke();
      }
    }
    const fr = tweenFrame(prev, curr, e);
    if (cfg.trails && !algo.trail) drawTrails(ctx, map, [...history, fr]);
    const pulse = 0.5 + 0.5 * Math.sin(performance.now() / 480);
    drawFrame(ctx, map, fr, pulse);
    stat.textContent = `iter ${algo.iter} · f* = ${algo.best.toExponential(2)}`;
  }
  function refreshPanel() {
    C.spBody.innerHTML = algo.info().map(([k, v]) => `<div class="sp-row"><span>${k}</span><b>${v}</b></div>`).join("");
    narr.textContent = algo.status();
  }
  function advance() {
    ticks++;
    for (let s = 0; s < (algo.stepsPerTick || 1); s++) algo.step();
    history.push(curr); if (history.length > TRAIL) history.shift();
    prev = curr; curr = algo.frame(); tStart = performance.now(); refreshPanel();
  }

  C.sel.addEventListener("change", () => { rebuild(); refreshPanel(); });
  const rebuildKeys = cfg.rebuildOn || ["n"];
  for (const id in C.inputs) C.inputs[id].addEventListener("input", () => { if (rebuildKeys.includes(id)) { rebuild(); refreshPanel(); } else algo.params = readParams(); });
  C.spdSel.addEventListener("change", () => { tickMs = SPEEDS[C.spdSel.value]; });
  C.play.addEventListener("click", () => { running = !running; C.play.textContent = running ? "❚❚" : "▶"; if (running) tStart = performance.now(); });
  C.step.addEventListener("click", () => { if (running) { running = false; C.play.textContent = "▶"; } advance(); paint(1); });
  C.reset.addEventListener("click", () => { rebuild(); refreshPanel(); });

  function loop(now) {
    if (running && inView && algo) {
      if (finishing) {
        paint(1);
        if (now >= restartAt) { rebuild(); refreshPanel(); }
      } else {
        let t = (now - tStart) / tickMs;
        if (t >= 1) { advance(); t = 0; }
        paint(easeInOut(Math.min(1, t)));
        if (ticks >= LOOP_TICKS) { finishing = true; restartAt = now + 1100; }
      }
    }
    requestAnimationFrame(loop);
  }
  // Start fresh the first time the demo scrolls into view; pause when it leaves.
  const io = new IntersectionObserver((es) => es.forEach((e) => {
    inView = e.isIntersecting;
    if (inView && !started) { started = true; rebuild(); refreshPanel(); }
  }), { threshold: 0.3 });
  io.observe(root);
  let rt; window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => { rebuild(); refreshPanel(); }, 200); });
  rebuild(); refreshPanel(); requestAnimationFrame(loop);
}

// ---------------- Bayesian (1-D, step-driven) ----------------
function mountBayes(root) {
  root.innerHTML = "";
  const head = document.createElement("div"); head.className = "demo-head";
  head.innerHTML = `<span class="title">bayesian optimization · GP + expected improvement</span><span class="stat">·</span>`;
  const stat = head.querySelector(".stat");
  const grid = document.createElement("div"); grid.className = "demo-grid";
  const col = document.createElement("div"); col.className = "canvas-col";
  const cw = document.createElement("div"); cw.className = "canvas-wrap";
  const canvas = document.createElement("canvas"); cw.appendChild(canvas);
  const narr = document.createElement("div"); narr.className = "narration";
  narr.textContent = "Each step: fit the GP posterior, maximize EI(x), evaluate f there, repeat.";
  col.append(cw, narr);

  const wrap = document.createElement("div"); wrap.className = "controls";
  wrap.innerHTML = `
    <div class="ctrl"><label><span>length scale  ℓ</span><span class="val" id="bls">0.12</span></label>
      <input type="range" id="len" min="0.03" max="0.4" step="0.01" value="0.12">
      <div class="hint">RBF kernel smoothness prior</div></div>
    <div class="ctrl"><label><span>exploration  ξ</span><span class="val" id="bxi">0.01</span></label>
      <input type="range" id="xi" min="0" max="0.2" step="0.005" value="0.01">
      <div class="hint">EI margin: higher explores more</div></div>`;
  const btns = document.createElement("div"); btns.className = "btns";
  const sampleBtn = document.createElement("button"); sampleBtn.className = "btn"; sampleBtn.textContent = "sample next →";
  const autoBtn = document.createElement("button"); autoBtn.className = "btn ghost"; autoBtn.textContent = "▶ auto";
  const resetBtn = document.createElement("button"); resetBtn.className = "btn ghost"; resetBtn.textContent = "↻";
  btns.append(sampleBtn, autoBtn, resetBtn); wrap.appendChild(btns);
  const legend = document.createElement("div"); legend.className = "legend";
  legend.innerHTML = `<span><span class="dot" style="background:#9aa6b2"></span>true f (hidden)</span>
    <span><span class="dot" style="background:#5ec8ff"></span>GP mean μ</span>
    <span><span class="dot" style="background:#f0b72f"></span>observations</span>
    <span><span class="dot" style="background:#56d364"></span>EI acquisition</span>`;
  wrap.appendChild(legend);
  grid.append(col, wrap);
  root.append(head, grid);

  let algo, ctx, W, H, auto = false, seed = 7, frame = 0, inView = false, started = false, restartAt = 0;
  const MAX_STEPS = 9;
  function rebuild() {
    const d = setupCanvas(canvas); ctx = d.ctx; W = d.W; H = d.H;
    algo = createBayes({ seed, length: parseFloat(wrap.querySelector("#len").value), xi: parseFloat(wrap.querySelector("#xi").value) });
    render();
  }
  function render() { algo.draw(ctx, W, H); stat.textContent = `${algo.iter + 2} evals · f* = ${algo.best.toFixed(4)}`; }
  wrap.querySelector("#len").addEventListener("input", (e) => { wrap.querySelector("#bls").textContent = (+e.target.value).toFixed(2); algo.params.length = +e.target.value; algo.onParam(); render(); });
  wrap.querySelector("#xi").addEventListener("input", (e) => { wrap.querySelector("#bxi").textContent = (+e.target.value).toFixed(3); algo.params.xi = +e.target.value; algo.onParam(); render(); });
  sampleBtn.addEventListener("click", () => { algo.step(); render(); });
  autoBtn.addEventListener("click", () => { auto = !auto; autoBtn.textContent = auto ? "❚❚ stop" : "▶ auto"; });
  resetBtn.addEventListener("click", () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; rebuild(); });
  let rt; window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(rebuild, 200); });
  // Auto-run only while on screen; the first time it scrolls in, start sampling.
  const io = new IntersectionObserver((es) => es.forEach((e) => {
    inView = e.isIntersecting;
    if (inView && !started) { started = true; auto = true; autoBtn.textContent = "❚❚ stop"; }
  }), { threshold: 0.3 });
  io.observe(root);
  function loop(now) {
    if (auto && inView) {
      if (restartAt) {
        if (now >= restartAt) { restartAt = 0; seed = (seed * 1103515245 + 12345) & 0x7fffffff; rebuild(); }
      } else {
        frame++;
        if (frame % 40 === 0) { algo.step(); render(); if (algo.iter >= MAX_STEPS) restartAt = now + 1300; }
      }
    }
    requestAnimationFrame(loop);
  }
  rebuild(); requestAnimationFrame(loop);
}

function boot() {
  document.querySelectorAll(".demo[data-demo]").forEach((el) => {
    const key = el.getAttribute("data-demo");
    if (key === "trap") mountTrap(el);
    else if (key === "bayes") mountBayes(el);
    else if (CONFIGS[key]) mount2D(el, key);
  });
  const links = [...document.querySelectorAll("nav.toc a")];
  const map = new Map(links.map((a) => [a.getAttribute("href").slice(1), a]));
  const obs = new IntersectionObserver((es) => es.forEach((e) => {
    if (e.isIntersecting) { links.forEach((a) => a.classList.remove("active")); const a = map.get(e.target.id); if (a) a.classList.add("active"); }
  }), { rootMargin: "-40% 0px -55% 0px" });
  document.querySelectorAll("section[id]").forEach((s) => obs.observe(s));
}

if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
else boot();
