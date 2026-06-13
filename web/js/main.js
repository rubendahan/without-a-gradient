// Wires the demo widgets into the page: builds controls, renders the landscape,
// and runs the animation loop. Each <div class="demo" data-demo="..."> in the
// HTML is filled in here.
import { FUNCS, makeMapper, renderLandscape, setupCanvas } from "./landscape.js";
import { createPSO, createGA, createDE, createSA } from "./algos.js";
import { createBayes } from "./bayes.js";

// -------- shared control factory --------
function slider(id, label, min, max, step, value, fmt) {
  return { id, label, min, max, step, value, fmt: fmt || ((v) => v) };
}

const CONFIGS = {
  pso: {
    title: "particle swarm · live",
    factory: createPSO,
    funcKeys: ["rastrigin", "ackley", "himmelblau", "rosenbrock"],
    defaults: { n: 30, w: 0.7, c1: 1.5, c2: 1.5 },
    speed: 1,
    controls: [
      slider("n", "particles", 5, 80, 1, 30, (v) => v | 0),
      slider("w", "inertia w", 0, 1.2, 0.01, 0.7, (v) => v.toFixed(2)),
      slider("c1", "cognitive c₁", 0, 3, 0.05, 1.5, (v) => v.toFixed(2)),
      slider("c2", "social c₂", 0, 3, 0.05, 1.5, (v) => v.toFixed(2)),
    ],
    legend: [["#58a6ff", "particle + velocity"], ["#ffffff", "global best"]],
  },
  ga: {
    title: "genetic algorithm · live",
    factory: createGA,
    funcKeys: ["rastrigin", "ackley", "himmelblau", "sphere"],
    defaults: { n: 50, mut: 0.08, k: 3 },
    speed: 1,
    controls: [
      slider("n", "population", 10, 120, 2, 50, (v) => v | 0),
      slider("mut", "mutation σ", 0.005, 0.3, 0.005, 0.08, (v) => v.toFixed(3)),
      slider("k", "tournament k", 2, 8, 1, 3, (v) => v | 0),
    ],
    legend: [["#56d364", "fit individual"], ["#d2a8ff", "less fit"], ["#ffffff", "best"]],
  },
  de: {
    title: "differential evolution · live",
    factory: createDE,
    funcKeys: ["rastrigin", "ackley", "rosenbrock", "himmelblau"],
    defaults: { n: 40, F: 0.7, CR: 0.9 },
    speed: 1,
    controls: [
      slider("n", "population", 8, 90, 1, 40, (v) => v | 0),
      slider("F", "weight F", 0.1, 1.2, 0.05, 0.7, (v) => v.toFixed(2)),
      slider("CR", "crossover CR", 0.1, 1, 0.05, 0.9, (v) => v.toFixed(2)),
    ],
    legend: [["#d2a8ff", "candidate"], ["#ffffff", "best"]],
  },
  sa: {
    title: "simulated annealing · live",
    factory: createSA,
    funcKeys: ["rastrigin", "ackley", "himmelblau", "sphere"],
    defaults: { T0: 4, cool: 0.99 },
    speed: 2,
    controls: [
      slider("T0", "start temp T₀", 0.2, 12, 0.1, 4, (v) => v.toFixed(1)),
      slider("cool", "cooling α", 0.9, 0.999, 0.001, 0.99, (v) => v.toFixed(3)),
    ],
    legend: [["#f0b72f", "walker / accepted"], ["#ff7b72", "rejected"], ["#ffffff", "best"]],
    extraStat: (algo) => `T=${algo.temp.toFixed(3)}`,
  },
};

function buildControlsDOM(cfg) {
  const wrap = document.createElement("div");
  wrap.className = "controls";

  // function selector
  const fnCtrl = document.createElement("div");
  fnCtrl.className = "ctrl";
  const fnLabel = document.createElement("label");
  fnLabel.innerHTML = `<span>landscape</span>`;
  const sel = document.createElement("select");
  cfg.funcKeys.forEach((k) => {
    const o = document.createElement("option");
    o.value = k; o.textContent = `${FUNCS[k].label} — ${FUNCS[k].blurb}`;
    sel.appendChild(o);
  });
  fnCtrl.appendChild(fnLabel); fnCtrl.appendChild(sel);
  wrap.appendChild(fnCtrl);

  const inputs = {};
  for (const c of cfg.controls) {
    const ctrl = document.createElement("div");
    ctrl.className = "ctrl";
    const lab = document.createElement("label");
    const val = document.createElement("span"); val.className = "val";
    lab.innerHTML = `<span>${c.label}</span>`; lab.appendChild(val);
    const inp = document.createElement("input");
    inp.type = "range"; inp.min = c.min; inp.max = c.max; inp.step = c.step; inp.value = c.value;
    const sync = () => { val.textContent = c.fmt(parseFloat(inp.value)); };
    inp.addEventListener("input", sync); sync();
    ctrl.appendChild(lab); ctrl.appendChild(inp);
    wrap.appendChild(ctrl);
    inputs[c.id] = inp;
  }

  const btns = document.createElement("div");
  btns.className = "btns";
  const play = document.createElement("button"); play.className = "btn"; play.textContent = "❚❚ pause";
  const step = document.createElement("button"); step.className = "btn ghost"; step.textContent = "step";
  const reset = document.createElement("button"); reset.className = "btn ghost"; reset.textContent = "reset";
  btns.appendChild(play); btns.appendChild(step); btns.appendChild(reset);
  wrap.appendChild(btns);

  const legend = document.createElement("div");
  legend.className = "legend";
  for (const [color, text] of cfg.legend) {
    const s = document.createElement("span");
    s.innerHTML = `<span class="dot" style="background:${color}"></span>${text}`;
    legend.appendChild(s);
  }
  wrap.appendChild(legend);

  return { wrap, sel, inputs, play, step, reset };
}

function mount2D(root, key) {
  const cfg = CONFIGS[key];
  root.innerHTML = "";
  // head
  const head = document.createElement("div"); head.className = "demo-head";
  head.innerHTML = `<span class="title">${cfg.title}</span><span class="stat">—</span>`;
  const stat = head.querySelector(".stat");
  // body
  const body = document.createElement("div"); body.className = "demo-body";
  const cw = document.createElement("div"); cw.className = "canvas-wrap";
  const canvas = document.createElement("canvas"); cw.appendChild(canvas);
  body.appendChild(cw);
  const { wrap, sel, inputs, play, step, reset } = buildControlsDOM(cfg);
  body.appendChild(wrap);
  root.appendChild(head); root.appendChild(body);

  let running = true, algo, fn, map, bg, ctx, W, H;

  function readParams() {
    const p = { ...cfg.defaults };
    for (const id in inputs) p[id] = parseFloat(inputs[id].value);
    return p;
  }
  function rebuild() {
    fn = FUNCS[sel.value];
    const dim = setupCanvas(canvas);
    ctx = dim.ctx; W = dim.W; H = dim.H;
    bg = renderLandscape(fn, W, H);
    map = makeMapper(fn, W, H);
    algo = cfg.factory(fn, readParams());
  }
  function softParams() { if (algo) algo.params = readParams(); }

  sel.addEventListener("change", rebuild);
  for (const id in inputs) {
    inputs[id].addEventListener("input", () => {
      // particle/pop count changes need a rebuild; weights are live.
      if (id === "n") rebuild(); else softParams();
    });
  }
  play.addEventListener("click", () => { running = !running; play.textContent = running ? "❚❚ pause" : "▶ play"; });
  step.addEventListener("click", () => { if (algo) { algo.step(); render(); } });
  reset.addEventListener("click", rebuild);

  function render() {
    ctx.clearRect(0, 0, W, H);
    ctx.drawImage(bg, 0, 0, W, H);
    algo.draw(ctx, map, W, H);
    let s = `iter ${algo.iter}   best f = ${algo.best.toExponential(2)}`;
    if (cfg.extraStat) s += "   " + cfg.extraStat(algo);
    stat.textContent = s;
  }
  function loop() {
    if (running && algo) {
      for (let i = 0; i < (cfg.speed || 1); i++) algo.step();
      render();
    }
    requestAnimationFrame(loop);
  }

  // (re)build on resize so canvas stays crisp
  let rt;
  window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(rebuild, 200); });
  rebuild(); render(); loop();
}

function mountBayes(root) {
  root.innerHTML = "";
  const head = document.createElement("div"); head.className = "demo-head";
  head.innerHTML = `<span class="title">bayesian optimization · 1-D</span><span class="stat">—</span>`;
  const stat = head.querySelector(".stat");
  const body = document.createElement("div"); body.className = "demo-body";
  const cw = document.createElement("div"); cw.className = "canvas-wrap";
  const canvas = document.createElement("canvas"); cw.appendChild(canvas);
  body.appendChild(cw);

  const wrap = document.createElement("div"); wrap.className = "controls";
  wrap.innerHTML = `
    <div class="ctrl"><label><span>length scale</span><span class="val" id="bls">0.12</span></label>
      <input type="range" id="len" min="0.03" max="0.4" step="0.01" value="0.12"></div>
    <div class="ctrl"><label><span>exploration ξ</span><span class="val" id="bxi">0.01</span></label>
      <input type="range" id="xi" min="0" max="0.2" step="0.005" value="0.01"></div>`;
  const btns = document.createElement("div"); btns.className = "btns";
  const sampleBtn = document.createElement("button"); sampleBtn.className = "btn"; sampleBtn.textContent = "sample next →";
  const autoBtn = document.createElement("button"); autoBtn.className = "btn ghost"; autoBtn.textContent = "▶ auto";
  const resetBtn = document.createElement("button"); resetBtn.className = "btn ghost"; resetBtn.textContent = "reset";
  btns.appendChild(sampleBtn); btns.appendChild(autoBtn); btns.appendChild(resetBtn);
  wrap.appendChild(btns);
  const legend = document.createElement("div"); legend.className = "legend";
  legend.innerHTML = `<span><span class="dot" style="background:#f0b72f"></span>evaluations</span>
    <span><span class="dot" style="background:#58a6ff"></span>GP mean</span>
    <span><span class="dot" style="background:#56d364"></span>acquisition</span>`;
  wrap.appendChild(legend);
  body.appendChild(wrap);
  root.appendChild(head); root.appendChild(body);

  let algo, ctx, W, H, auto = false, seed = 7, frame = 0;
  function rebuild() {
    const dim = setupCanvas(canvas); ctx = dim.ctx; W = dim.W; H = dim.H;
    algo = createBayes({ seed, length: parseFloat(wrap.querySelector("#len").value), xi: parseFloat(wrap.querySelector("#xi").value) });
    render();
  }
  function render() {
    algo.draw(ctx, W, H);
    stat.textContent = `evals ${algo.iter + 2}   best f = ${algo.best.toFixed(4)}`;
  }
  wrap.querySelector("#len").addEventListener("input", (e) => { wrap.querySelector("#bls").textContent = (+e.target.value).toFixed(2); algo.params.length = +e.target.value; algo.onParam(); render(); });
  wrap.querySelector("#xi").addEventListener("input", (e) => { wrap.querySelector("#bxi").textContent = (+e.target.value).toFixed(3); algo.params.xi = +e.target.value; algo.onParam(); render(); });
  sampleBtn.addEventListener("click", () => { algo.step(); render(); });
  autoBtn.addEventListener("click", () => { auto = !auto; autoBtn.textContent = auto ? "❚❚ stop" : "▶ auto"; });
  resetBtn.addEventListener("click", () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; rebuild(); });
  let rt; window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(rebuild, 200); });

  function loop() { if (auto) { frame++; if (frame % 30 === 0) { algo.step(); render(); } } requestAnimationFrame(loop); }
  rebuild(); loop();
}

// -------- boot --------
function boot() {
  document.querySelectorAll(".demo[data-demo]").forEach((el) => {
    const key = el.getAttribute("data-demo");
    if (key === "bayes") mountBayes(el);
    else if (CONFIGS[key]) mount2D(el, key);
  });

  // scroll-spy for the TOC
  const links = [...document.querySelectorAll("nav.toc a")];
  const map = new Map(links.map((a) => [a.getAttribute("href").slice(1), a]));
  const obs = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (e.isIntersecting) {
        links.forEach((a) => a.classList.remove("active"));
        const a = map.get(e.target.id); if (a) a.classList.add("active");
      }
    });
  }, { rootMargin: "-40% 0px -55% 0px" });
  document.querySelectorAll("section[id]").forEach((s) => obs.observe(s));
}

if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
else boot();
