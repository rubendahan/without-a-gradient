import { setupCanvas } from "./landscape.js";
import {
  DOMAIN,
  STARTS,
  objective,
  descend,
  MINIMUM,
  pointAt,
} from "./descent.js";

const BLUE = "#2864a5",
  GREEN = "#278362",
  RED = "#bc593d";
const DURATION = 12500;

export function mountTrap(root) {
  root.classList.add("descent-demo");
  root.innerHTML = `<div class="demo-head"><span class="title">Gradient descent</span><span class="stat"></span></div>
    <div class="descent-toolbar"><span>Same objective · same step size · different starts</span><div class="btns"><button class="btn descent-play">Play</button><button class="btn ghost descent-step">Step</button><button class="btn ghost descent-reset">Reset</button></div></div>
    <div class="descent-landscape"><canvas tabindex="0" role="img" aria-label="Gradient descent on a one-dimensional objective. Click to add a starting point, or use arrow keys and Enter."></canvas></div>
    <div class="descent-legend"><span><i class="descent-hollow"></i>Starting point</span><span><i style="background:${BLUE}"></i>Descending</span><span><i style="background:${GREEN}"></i>Lowest minimum</span><span><i style="background:${RED}"></i>Other local minimum</span></div>
    <div class="descent-loss"><div class="descent-loss-heading"><strong>Loss for each starting point</strong><span>Same iterations as the landscape above</span></div><canvas role="img" aria-label="Objective values for all starting points against iteration"></canvas></div>
    <div class="descent-bottom"><div class="descent-outcome" role="status"></div><span>Click the landscape to add a start and replay.</span></div>`;
  const canvas = root.querySelector(".descent-landscape canvas"),
    loss = root.querySelector(".descent-loss canvas");
  const play = root.querySelector(".descent-play"),
    step = root.querySelector(".descent-step");
  const stat = root.querySelector(".stat"),
    outcome = root.querySelector(".descent-outcome");
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  let starts = [...STARTS],
    runs = [],
    maxIteration = 1,
    iteration = 0;
  let running = false,
    singleStep = null,
    inView = false,
    seen = false,
    lastTime = 0,
    elapsed = 0,
    hold = 0;
  let pointer = null,
    keyboard = false,
    lastSummary = "";
  let plot, lossPlot, background, lossBackground;
  let low = MINIMUM.value - 0.5,
    high = -Infinity;
  for (let i = 0; i <= 800; i++)
    high = Math.max(
      high,
      objective(DOMAIN.lo + (i / 800) * (DOMAIN.hi - DOMAIN.lo)),
    );
  high += 0.45;
  const finished = (run) => iteration >= run.path.length - 1;
  const reachedLowest = (run) =>
    run.converged && Math.abs(run.rest - MINIMUM.x) < 0.02;
  const colour = (run) =>
    !finished(run) ? BLUE : reachedLowest(run) ? GREEN : RED;
  const iterationAt = (time) =>
    Math.expm1(Math.log1p(maxIteration) * Math.min(1, time / DURATION));
  const timeAt = (value) =>
    (DURATION * Math.log1p(value)) / Math.log1p(maxIteration);

  function sync() {
    const done = iteration >= maxIteration;
    play.textContent = running ? "Pause" : done ? "Replay" : "Play";
    play.setAttribute("aria-pressed", String(running));
    step.disabled = done;
    root.dataset.state = done
      ? "complete"
      : running
        ? "playing"
        : singleStep
          ? "stepping"
          : "paused";
    root.dataset.iteration = String(Math.floor(iteration + 1e-7));
    const settled = runs.filter(finished),
      found = settled.filter(reachedLowest).length;
    stat.textContent = `Iteration ${Math.floor(iteration + 1e-7)} · ${runs.length} starts`;
    const summary = done
      ? `${found} of ${runs.length} starts reached the lowest minimum. The others settled in local minima.`
      : iteration === 0
        ? "The hollow circles mark the starting points. Watch where each descent stops."
        : `${settled.length} of ${runs.length} descents have settled. Every point follows the same update.`;
    if (summary !== lastSummary) {
      outcome.textContent = summary;
      lastSummary = summary;
    }
    canvas.setAttribute(
      "aria-label",
      `Gradient descent, iteration ${Math.floor(iteration)}. ${settled.length} of ${runs.length} starts settled, ${found} at the lowest minimum. Click to add a start; arrow keys choose a position and Enter adds it.`,
    );
  }
  function prepare() {
    runs = starts.map(descend);
    maxIteration = Math.max(1, ...runs.map((r) => r.path.length - 1));
    iteration = 0;
    elapsed = 0;
    running = false;
    singleStep = null;
    hold = 0;
    if (plot) resize();
    sync();
  }
  function begin() {
    seen = true;
    if (iteration >= maxIteration) prepare();
    singleStep = null;
    running = true;
    elapsed = timeAt(iteration);
    hold = iteration === 0 ? 650 : 0;
    lastTime = performance.now();
    sync();
  }
  function addStart(x) {
    if (starts.length >= 20) starts.pop();
    starts.push(x);
    prepare();
    if (!reduced) begin();
    draw();
  }
  play.addEventListener("click", () => {
    seen = true;
    if (running || singleStep) {
      running = false;
      singleStep = null;
      sync();
    } else begin();
  });
  step.addEventListener("click", () => {
    seen = true;
    running = false;
    hold = 0;
    const target = Math.min(maxIteration, Math.floor(iteration + 1e-7) + 1);
    if (reduced) {
      iteration = target;
      singleStep = null;
      elapsed = timeAt(iteration);
      draw();
    } else singleStep = { from: iteration, to: target, elapsed: 0 };
    sync();
  });
  root.querySelector(".descent-reset").addEventListener("click", () => {
    seen = true;
    starts = [...STARTS];
    prepare();
    pointer = null;
    draw();
  });
  const xFromEvent = (event) => {
    const box = canvas.getBoundingClientRect();
    return Math.max(
      DOMAIN.lo,
      Math.min(
        DOMAIN.hi,
        DOMAIN.lo +
          ((event.clientX - box.left - plot.left) / plot.width) *
            (DOMAIN.hi - DOMAIN.lo),
      ),
    );
  };
  canvas.addEventListener("pointermove", (event) => {
    if (event.pointerType !== "touch") {
      pointer = xFromEvent(event);
      keyboard = false;
      if (!running) draw();
    }
  });
  canvas.addEventListener("pointerleave", () => {
    if (!keyboard) pointer = null;
    if (!running) draw();
  });
  canvas.addEventListener("click", (event) => {
    pointer = null;
    addStart(xFromEvent(event));
  });
  canvas.addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight", "Enter", " "].includes(event.key)) return;
    event.preventDefault();
    keyboard = true;
    if (event.key === "Enter" || event.key === " ") {
      addStart(pointer ?? 0);
      pointer = null;
    } else {
      pointer = Math.max(
        DOMAIN.lo,
        Math.min(
          DOMAIN.hi,
          (pointer ?? 0) + (event.key === "ArrowRight" ? 0.15 : -0.15),
        ),
      );
      draw();
    }
  });

  function axes(target, isLoss) {
    const { ctx, W, H } = target;
    const left = W < 500 ? 38 : 52,
      right = W < 500 ? 20 : 28;
    const top = isLoss ? 18 : 27,
      bottom = isLoss ? 31 : 37;
    const width = W - left - right,
      height = H - top - bottom;
    return {
      ctx,
      W,
      H,
      left,
      top,
      width,
      height,
      x: (value) =>
        left +
        (isLoss
          ? value / maxIteration
          : (value - DOMAIN.lo) / (DOMAIN.hi - DOMAIN.lo)) *
          width,
      y: (value) => top + ((high - value) / (high - low)) * height,
    };
  }
  function basePlot(p, isLoss) {
    const bg = document.createElement("canvas");
    bg.width = Math.round(p.W * devicePixelRatio);
    bg.height = Math.round(p.H * devicePixelRatio);
    const c = bg.getContext("2d");
    c.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
    c.fillStyle = "#fff";
    c.fillRect(0, 0, p.W, p.H);
    c.font = "11px Segoe UI, sans-serif";
    c.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const val = low + 0.15 + (i / 4) * (high - low - 0.3),
        y = p.y(val);
      c.strokeStyle = "#e9eef3";
      c.beginPath();
      c.moveTo(p.left, y);
      c.lineTo(p.left + p.width, y);
      c.stroke();
      c.fillStyle = "#63758a";
      c.textAlign = "right";
      c.fillText(val.toFixed(1), p.left - 9, y + 4);
      const xx = isLoss
        ? (i / 4) * maxIteration
        : DOMAIN.lo + (i / 4) * (DOMAIN.hi - DOMAIN.lo);
      c.textAlign = "center";
      c.fillText(
        isLoss ? Math.round(xx).toString() : xx.toFixed(1),
        p.x(xx),
        p.H - 13,
      );
    }
    c.strokeStyle = "#27836255";
    c.setLineDash([4, 5]);
    c.beginPath();
    c.moveTo(p.left, p.y(MINIMUM.value));
    c.lineTo(p.left + p.width, p.y(MINIMUM.value));
    c.stroke();
    c.setLineDash([]);
    c.fillStyle = "#63758a";
    c.textAlign = "left";
    c.fillText("f(x)", p.left, 13);
    c.textAlign = "right";
    c.fillText(isLoss ? "iteration" : "x", p.W - 5, isLoss ? 13 : p.H - 13);
    if (!isLoss) {
      c.beginPath();
      c.moveTo(p.x(DOMAIN.lo), p.top + p.height);
      for (let i = 0; i <= 900; i++) {
        const x = DOMAIN.lo + (i / 900) * (DOMAIN.hi - DOMAIN.lo);
        c.lineTo(p.x(x), p.y(objective(x)));
      }
      c.lineTo(p.x(DOMAIN.hi), p.top + p.height);
      c.closePath();
      c.fillStyle = "#2864a507";
      c.fill();
      c.beginPath();
      for (let i = 0; i <= 900; i++) {
        const x = DOMAIN.lo + (i / 900) * (DOMAIN.hi - DOMAIN.lo);
        i
          ? c.lineTo(p.x(x), p.y(objective(x)))
          : c.moveTo(p.x(x), p.y(objective(x)));
      }
      c.strokeStyle = "#527a9d";
      c.lineWidth = 2.1;
      c.stroke();
      c.fillStyle = "#278362";
      c.textAlign = "left";
      c.font = "11px Segoe UI,sans-serif";
      c.fillText(
        "Lowest minimum",
        Math.min(p.W - 110, p.x(MINIMUM.x) + 13),
        p.y(MINIMUM.value) + 21,
      );
      c.strokeStyle = "#278362";
      c.lineWidth = 1.4;
      c.beginPath();
      c.arc(p.x(MINIMUM.x), p.y(MINIMUM.value), 8, 0, 2 * Math.PI);
      c.stroke();
    }
    return bg;
  }
  function resize() {
    plot = axes(setupCanvas(canvas), false);
    lossPlot = axes(setupCanvas(loss), true);
    background = basePlot(plot, false);
    lossBackground = basePlot(lossPlot, true);
    draw();
  }
  function draw() {
    if (!plot) return;
    const c = plot.ctx,
      l = lossPlot.ctx;
    c.drawImage(background, 0, 0, plot.W, plot.H);
    l.drawImage(lossBackground, 0, 0, lossPlot.W, lossPlot.H);
    const buckets = new Map();
    for (const run of runs) {
      const x = pointAt(run, iteration),
        limit = Math.min(Math.floor(iteration), run.path.length - 1),
        col = colour(run);
      // Draw the visited part of the objective, including interpolated motion on the curve.
      c.beginPath();
      c.moveTo(plot.x(run.start), plot.y(objective(run.start)));
      for (let i = 1; i <= limit + 1; i++) {
        const a = run.path[Math.min(i - 1, limit)],
          b = i > limit ? x : run.path[i];
        for (let j = 1; j <= 6; j++) {
          const z = a + ((b - a) * j) / 6;
          c.lineTo(plot.x(z), plot.y(objective(z)));
        }
      }
      c.strokeStyle = col;
      c.globalAlpha = finished(run) ? 0.48 : 0.32;
      c.lineWidth = 3;
      c.stroke();
      c.globalAlpha = 1;
      c.strokeStyle = "#96a7b8";
      c.fillStyle = "white";
      c.lineWidth = 1.3;
      c.beginPath();
      c.arc(
        plot.x(run.start),
        plot.y(objective(run.start)),
        3.8,
        0,
        2 * Math.PI,
      );
      c.fill();
      c.stroke();
      if (iteration > 0) {
        c.fillStyle = col + "13";
        c.beginPath();
        c.arc(plot.x(x), plot.y(objective(x)), 10, 0, 2 * Math.PI);
        c.fill();
        c.fillStyle = col;
        c.strokeStyle = "white";
        c.lineWidth = 1.7;
        c.beginPath();
        c.arc(plot.x(x), plot.y(objective(x)), 5.1, 0, 2 * Math.PI);
        c.fill();
        c.stroke();
      }
      l.strokeStyle = col;
      l.lineWidth = finished(run) ? 1.8 : 1.4;
      l.globalAlpha = finished(run) ? 0.85 : 0.6;
      l.beginPath();
      run.path.slice(0, limit + 1).forEach((v, i) => {
        i
          ? l.lineTo(lossPlot.x(i), lossPlot.y(objective(v)))
          : l.moveTo(lossPlot.x(i), lossPlot.y(objective(v)));
      });
      if (iteration > limit && !finished(run))
        l.lineTo(lossPlot.x(iteration), lossPlot.y(objective(x)));
      if (finished(run))
        l.lineTo(lossPlot.x(iteration), lossPlot.y(objective(run.rest)));
      l.stroke();
      l.globalAlpha = 1;
      l.fillStyle = col;
      l.beginPath();
      l.arc(
        lossPlot.x(iteration),
        lossPlot.y(objective(x)),
        2.3,
        0,
        Math.PI * 2,
      );
      l.fill();
      if (finished(run)) {
        const key = run.rest.toFixed(2);
        if (!buckets.has(key)) buckets.set(key, { x: run.rest, n: 0, col });
        buckets.get(key).n++;
      }
    }
    for (const b of buckets.values())
      if (b.n > 1) {
        const x = plot.x(b.x),
          y = plot.y(objective(b.x)) - 19;
        c.fillStyle = "white";
        c.beginPath();
        c.arc(x, y, 9, 0, 2 * Math.PI);
        c.fill();
        c.fillStyle = b.col;
        c.font = "600 11px Segoe UI,sans-serif";
        c.textAlign = "center";
        c.fillText(String(b.n), x, y + 4);
      }
    if (pointer !== null) {
      const x = plot.x(pointer),
        y = plot.y(objective(pointer));
      c.strokeStyle = "#285b9266";
      c.setLineDash([3, 4]);
      c.beginPath();
      c.moveTo(x, plot.top);
      c.lineTo(x, plot.top + plot.height);
      c.stroke();
      c.setLineDash([]);
      c.strokeStyle = BLUE;
      c.lineWidth = 1.5;
      c.beginPath();
      c.arc(x, y, 7, 0, Math.PI * 2);
      c.stroke();
    }
    sync();
  }
  new ResizeObserver(resize).observe(canvas.parentElement);
  new ResizeObserver(resize).observe(loss.parentElement);
  new IntersectionObserver(
    (entries) => {
      inView = entries[0].isIntersecting;
      lastTime = performance.now();
      if (inView && !seen) {
        seen = true;
        if (!reduced) begin();
      }
    },
    { threshold: 0.25 },
  ).observe(root);
  document.addEventListener("visibilitychange", () => {
    lastTime = performance.now();
  });
  function frame(now) {
    const dt = Math.min(80, Math.max(0, now - lastTime));
    lastTime = now;
    if (inView && !document.hidden) {
      if (running) {
        if (hold > 0) hold = Math.max(0, hold - dt);
        else {
          elapsed = Math.min(DURATION, elapsed + dt);
          iteration = elapsed >= DURATION ? maxIteration : iterationAt(elapsed);
          if (iteration >= maxIteration) running = false;
        }
        draw();
      } else if (singleStep) {
        singleStep.elapsed += dt;
        const t = Math.min(1, singleStep.elapsed / 750),
          e = t * t * (3 - 2 * t);
        iteration = singleStep.from + (singleStep.to - singleStep.from) * e;
        if (t === 1) {
          iteration = singleStep.to;
          singleStep = null;
          elapsed = timeAt(iteration);
        }
        draw();
      }
    }
    requestAnimationFrame(frame);
  }
  prepare();
  resize();
  requestAnimationFrame(frame);
}
