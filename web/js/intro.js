// The hook. A 1-D bumpy landscape, drawn ONCE as a clear static figure. Several
// points start at fixed spots and do pure greedy descent; each trail is colored
// by where it ends up. Only the few that start inside the global basin (green)
// reach the true bottom; the rest settle into whichever local dip they were
// nearest (red). That stuck-ness is the whole reason metaheuristics exist.
import { setupCanvas, tag } from "./landscape.js";

const LO = -4.2,
  HI = 5.2;
// A hand-tuned wiggle with several dips of different depths.
function f(x) {
  return (
    0.85 * Math.cos(1.7 * x) +
    0.5 * Math.cos(3.6 * x + 1.1) +
    0.32 * Math.cos(6.1 * x + 0.4) +
    0.09 * x * x -
    0.15 * x
  );
}
function grad(x) {
  const h = 1e-3;
  return (f(x + h) - f(x - h)) / (2 * h);
}

// find global min by dense scan
let GX = LO,
  GMIN = Infinity;
for (let x = LO; x <= HI; x += 0.001) {
  const v = f(x);
  if (v < GMIN) {
    GMIN = v;
    GX = x;
  }
}

// Greedy descent from x0: return the resting point and the path it took.
function descend(x0) {
  let x = x0;
  const path = [x];
  for (let s = 0; s < 800; s++) {
    const gx = grad(x);
    if (Math.abs(gx) < 1e-4) break;
    x = Math.max(LO, Math.min(HI, x - 0.025 * gx));
    if (s % 3 === 0) path.push(x);
  }
  path.push(x);
  return { rest: x, path };
}

// Fixed start positions ; the story is the same every render.
const STARTS = Array.from(
  { length: 13 },
  (_, i) => LO + 0.35 + ((i + 0.5) / 13) * (HI - LO - 0.7),
);

export function mountTrap(root) {
  root.innerHTML = "";
  const head = document.createElement("div");
  head.className = "demo-head";
  head.innerHTML = `<span class="title">Local descent / different starts, different answers</span><span class="stat" id="ts">·</span>`;
  const stat = head.querySelector("#ts");
  const cw = document.createElement("div");
  cw.className = "canvas-wrap solo";
  const canvas = document.createElement("canvas");
  cw.appendChild(canvas);
  root.appendChild(head);
  root.appendChild(cw);

  let ctx, W, H;
  const starts = [...STARTS];
  canvas.tabIndex = 0;
  canvas.setAttribute("role", "img");
  canvas.setAttribute(
    "aria-label",
    "Local descent. Click to add a start, or use left and right arrow keys to choose a position and Enter to add it.",
  );
  let chosen = 0;
  function addStart(x) {
    if (starts.length >= 25) starts.splice(13, 1);
    starts.push(x);
    draw();
  }
  canvas.addEventListener("click", (e) => {
    const r = canvas.getBoundingClientRect();
    addStart(
      Math.max(
        LO,
        Math.min(
          HI,
          LO + ((e.clientX - r.left - PADX) / (W - 2 * PADX)) * (HI - LO),
        ),
      ),
    );
  });
  canvas.addEventListener("keydown", (e) => {
    if (["ArrowLeft", "ArrowRight", "Enter"].includes(e.key)) {
      e.preventDefault();
      if (e.key === "Enter") addStart(chosen);
      else
        chosen = Math.max(
          LO,
          Math.min(HI, chosen + (e.key === "ArrowRight" ? 0.2 : -0.2)),
        );
      canvas.setAttribute(
        "aria-label",
        `Selected start ${chosen.toFixed(1)}. Press Enter to add it.`,
      );
    }
  });
  const actions = document.createElement("div");
  actions.className = "demo-actions";
  actions.innerHTML =
    '<button class="btn ghost">Reset starting points</button>';
  actions.style.padding = "12px 20px";
  root.append(actions);
  actions.querySelector("button").addEventListener("click", () => {
    starts.splice(0, starts.length, ...STARTS);
    draw();
  });
  const PADX = 40,
    PADT = 28,
    PADB = 35;

  function ymm() {
    let lo = Infinity,
      hi = -Infinity;
    for (let x = LO; x <= HI; x += 0.01) {
      const v = f(x);
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
    }
    return [lo - 0.2, hi + 0.2];
  }
  let YLO, YHI;
  const X2 = (x) => PADX + ((x - LO) / (HI - LO)) * (W - 2 * PADX);
  const Y2 = (y) => PADT + (1 - (y - YLO) / (YHI - YLO)) * (H - PADT - PADB);

  function draw() {
    const d = setupCanvas(canvas);
    ctx = d.ctx;
    W = d.W;
    H = d.H;
    [YLO, YHI] = ymm();
    ctx.clearRect(0, 0, W, H);

    ctx.strokeStyle = "#e4e9ef";
    ctx.lineWidth = 1;
    ctx.fillStyle = "#637487";
    ctx.font = "10px Segoe UI, sans-serif";
    for (let i = 0; i <= 4; i++) {
      const y = YLO + (i / 4) * (YHI - YLO);
      ctx.beginPath();
      ctx.moveTo(PADX, Y2(y));
      ctx.lineTo(W - PADX, Y2(y));
      ctx.stroke();
      ctx.textAlign = "right";
      ctx.fillText(y.toFixed(1), PADX - 8, Y2(y) + 3);
      const x = LO + (i / 4) * (HI - LO);
      ctx.textAlign = "center";
      ctx.fillText(x.toFixed(1), X2(x), H - 13);
    }
    ctx.textAlign = "left";
    ctx.fillText("f(x)", PADX, 15);
    ctx.fillText("x", W - PADX + 16, H - 13);
    // fill under the curve with a cool gradient
    const g = ctx.createLinearGradient(0, PADT, 0, H - PADB);
    g.addColorStop(0, "rgba(100,147,183,0.02)");
    g.addColorStop(1, "rgba(100,147,183,0.10)");
    ctx.beginPath();
    ctx.moveTo(X2(LO), H - PADB);
    for (let x = LO; x <= HI; x += 0.02) ctx.lineTo(X2(x), Y2(f(x)));
    ctx.lineTo(X2(HI), H - PADB);
    ctx.closePath();
    ctx.fillStyle = g;
    ctx.fill();

    // the curve
    ctx.strokeStyle = "#376b8f";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    for (let x = LO; x <= HI; x += 0.02) {
      const px = X2(x),
        py = Y2(f(x));
      x === LO ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    }
    ctx.stroke();

    // run every descent, bucket by resting basin so stacked balls don't overlap
    const runs = starts.map((x0) => {
      const r = descend(x0);
      return { ...r, found: Math.abs(r.rest - GX) < 0.25 };
    });
    const buckets = new Map();
    for (const r of runs) {
      const key = Math.round(r.rest / 0.25);
      (buckets.get(key) || buckets.set(key, []).get(key)).push(r);
    }

    // faint descent trails first (under the markers)
    for (const r of runs) {
      ctx.strokeStyle = r.found ? "rgba(39,131,98,0.5)" : "rgba(195,89,56,0.4)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      r.path.forEach((x, i) => {
        const px = X2(x),
          py = Y2(f(x));
        i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
      });
      ctx.stroke();
    }

    // start markers (gray, hollow) at each release point
    for (const r of runs) {
      const px = X2(r.path[0]),
        py = Y2(f(r.path[0]));
      ctx.strokeStyle = "rgba(154,166,178,0.85)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(px, py, 3.5, 0, Math.PI * 2);
      ctx.stroke();
    }

    // the real bottom
    const gx = X2(GX),
      gy = Y2(GMIN);
    ctx.strokeStyle = "rgba(45,67,91,0.9)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(gx, gy, 8, 0, Math.PI * 2);
    ctx.stroke();
    tag(
      ctx,
      "global minimum",
      gx + 12,
      gy - 16,
      "rgba(255,255,255,0.95)",
      "#263e55",
    );

    // resting balls, fanned out within each basin so the count is legible
    for (const group of buckets.values()) {
      group.forEach((r, i) => {
        const off = (i - (group.length - 1) / 2) * 11;
        const px = X2(r.rest) + off,
          py = Y2(f(r.rest)) - 6;
        ctx.fillStyle = r.found ? "#278362" : "#c35938";
        ctx.save();
        ctx.shadowColor = "rgba(0,0,0,0.4)";
        ctx.shadowBlur = 4;
        ctx.beginPath();
        ctx.arc(px, py, 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      });
    }

    const found = runs.filter((r) => r.found).length;
    stat.textContent = `${found} of ${runs.length} starts reached the global minimum`;
  }

  let rt;
  window.addEventListener("resize", () => {
    clearTimeout(rt);
    rt = setTimeout(draw, 200);
  });
  draw();
}
