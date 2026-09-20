// Shared figures: a full-run loss trace and an equal-scale view of one operation.
const number = (value) =>
  value === 0
    ? "0"
    : Math.abs(value) >= 0.001 && Math.abs(value) < 10000
      ? Number(value.toPrecision(4)).toString()
      : value.toExponential(2);
export function renderLoss(svg, scores, progress = 1, linear = false) {
  const W = 360,
    H = 185,
    L = 48,
    R = 15,
    T = 15,
    B = 32;
  const maxX = Math.max(1, scores.at(-1).n);
  const transform = (v) => (linear ? v : Math.log10(1 + Math.max(0, v)));
  const inverse = (v) => (linear ? v : 10 ** v - 1);
  const minY = linear ? Math.min(...scores.map((s) => s.f)) - 0.05 : 0;
  const maxY = linear
    ? Math.max(...scores.map((s) => s.f)) + 0.05
    : Math.max(1e-9, ...scores.map((s) => transform(s.f)));
  const x = (v) => L + (v / maxX) * (W - L - R);
  const y = (v) =>
    T + (1 - (transform(v) - minY) / (maxY - minY)) * (H - T - B);
  const visible = scores.map((s) => ({ ...s }));
  if (visible.length > 1) {
    const a = visible.at(-2),
      b = visible.at(-1);
    b.n = a.n + (b.n - a.n) * progress;
    b.f = inverse(
      transform(a.f) + (transform(b.f) - transform(a.f)) * progress,
    );
  }
  const path = visible
    .map((s, i) => `${i ? "L" : "M"}${x(s.n).toFixed(2)},${y(s.f).toFixed(2)}`)
    .join(" ");
  const last = visible.at(-1),
    first = visible[0];
  svg.innerHTML =
    [0, 0.5, 1]
      .map((t) => {
        const value = inverse(minY + t * (maxY - minY)),
          yy = y(value);
        return `<line x1="${L}" x2="${W - R}" y1="${yy}" y2="${yy}" stroke="#e3eaf0" ${t ? 'stroke-dasharray="3 4"' : ""}/><text x="${L - 8}" y="${yy + 4}" text-anchor="end">${number(value)}</text>`;
      })
      .join("") +
    `<path d="${path} L${x(last.n)},${H - B} L${x(first.n)},${H - B} Z" fill="#2864a5" opacity=".055"/>
    <path d="${path}" fill="none" stroke="#2864a5" stroke-width="2.2" stroke-linejoin="round"/>
    <circle cx="${x(last.n)}" cy="${y(last.f)}" r="3.3" fill="#2864a5" stroke="white" stroke-width="1.5"/>
    <text x="${L}" y="${H - 14}">0</text><text x="${(L + W - R) / 2}" y="${H - 14}" text-anchor="middle">${Math.round(maxX / 2)}</text><text x="${W - R}" y="${H - 14}" text-anchor="end">${maxX}</text>`;
  svg.querySelectorAll("text").forEach((t) => {
    t.setAttribute("fill", "#63758a");
    t.setAttribute("font-size", "10");
    t.setAttribute("font-family", "Segoe UI, sans-serif");
  });
}

const DESCRIPTIONS = {
  pso: [
    "One particle",
    "The highlighted particle moves under inertia and attraction to its personal and shared records.",
  ],
  ga: [
    "One offspring",
    "Two selected parents produce an offspring through crossover and possible mutation.",
  ],
  de: [
    "One donor vector",
    "A scaled population difference is added to the base point before crossover and selection.",
  ],
  sa: [
    "One proposal",
    "Compare a proposed point with the current point; temperature controls uphill acceptance.",
  ],
  cmaes: [
    "One distribution update",
    "The previous sampling ellipse is dashed. The updated mean and covariance are shown in colour.",
  ],
};

export function renderOperation(
  svg,
  caption,
  key,
  previous,
  current,
  iteration,
  progress = 1,
) {
  const [title, text] = DESCRIPTIONS[key];
  caption.innerHTML = `<strong>${title}</strong><span>${iteration ? text : "Press Play to follow the full run, or Step to inspect one iteration."}</span>`;
  let rings = (current.rings || []).filter(
    (r) => r.label !== "best" && r.label !== "elite" && r.label !== "ĝ best",
  );
  let links = (current.links || []).filter((l) => !l.vel);
  let ellipses = [];
  if (key === "pso" && current.operation) {
    const o = current.operation;
    rings = [
      { x: o.x, y: o.y, color: "#71869b", label: "start" },
      { x: o.px, y: o.py, color: "#278362", label: "personal best" },
      { x: o.gx, y: o.gy, color: "#2864a5", label: "swarm best" },
      { x: o.nx, y: o.ny, color: "#ae7024", label: "next" },
    ];
    links = [
      { x1: o.x, y1: o.y, x2: o.px, y2: o.py, color: "#278362", dash: [3, 3] },
      { x1: o.x, y1: o.y, x2: o.gx, y2: o.gy, color: "#2864a5", dash: [3, 3] },
      { x1: o.x, y1: o.y, x2: o.nx, y2: o.ny, color: "#ae7024" },
    ];
  }
  if (key === "cmaes") {
    ellipses = [
      ...(previous.ellipses || []).slice(-1).map((e) => ({ ...e, old: true })),
      ...(current.ellipses || []).slice(-1),
    ];
    rings = current.rings.filter((r) => r.label === "mean m");
  }
  const all = [
    ...rings.map((r) => [r.x, r.y]),
    ...links.flatMap((l) => [
      [l.x1, l.y1],
      [l.x2, l.y2],
    ]),
  ];
  for (const e of ellipses) {
    const rx = Math.hypot(e.ax, e.bx),
      ry = Math.hypot(e.ay, e.by);
    all.push([e.cx - rx, e.cy - ry], [e.cx + rx, e.cy + ry]);
  }
  if (!all.length) all.push([-1, -1], [1, 1]);
  const xs = all.map((p) => p[0]),
    ys = all.map((p) => p[1]);
  const minX = Math.min(...xs),
    maxX = Math.max(...xs),
    minY = Math.min(...ys),
    maxY = Math.max(...ys);
  const scale = Math.min(
    230 / Math.max(1e-5, maxX - minX),
    100 / Math.max(1e-5, maxY - minY),
  );
  const project = (x, y) => [
    180 + (x - (minX + maxX) / 2) * scale,
    77 - (y - (minY + maxY) / 2) * scale,
  ];
  const short = (label) =>
    ({
      "particle xᵢ": "particle",
      "donor xₐ+F(x_b−x_c)": "donor",
      "xₐ (base)": "base",
      "mean m": "mean",
    })[label] ||
    label ||
    "";
  let content = `<path d="M20 142H340" stroke="#e6edf2"/>`;
  for (const e of ellipses) {
    const path = Array.from({ length: 65 }, (_, i) => {
      const t = (i / 64) * Math.PI * 2;
      const p = project(
        e.cx + Math.cos(t) * e.ax + Math.sin(t) * e.bx,
        e.cy + Math.cos(t) * e.ay + Math.sin(t) * e.by,
      );
      return `${i ? "L" : "M"}${p[0]},${p[1]}`;
    }).join(" ");
    content += `<path d="${path}Z" fill="${e.old ? "none" : "#287f850c"}" stroke="${e.old ? "#9baebb" : "#287f85"}" stroke-width="1.8" ${e.old ? 'stroke-dasharray="4 4"' : ""}/>`;
  }
  for (const l of links) {
    const a = project(l.x1, l.y1),
      dest = project(l.x2, l.y2),
      b = [
        a[0] + (dest[0] - a[0]) * progress,
        a[1] + (dest[1] - a[1]) * progress,
      ];
    const angle = Math.atan2(b[1] - a[1], b[0] - a[0]);
    content += `<path d="M${a}L${b}" fill="none" stroke="${l.color}" stroke-width="1.8" ${l.dash ? 'stroke-dasharray="4 4"' : ""}/>`;
    if (Math.hypot(b[0] - a[0], b[1] - a[1]) > 12)
      content += `<path d="M${b[0] - 6 * Math.cos(angle - 0.45)},${b[1] - 6 * Math.sin(angle - 0.45)}L${b}L${b[0] - 6 * Math.cos(angle + 0.45)},${b[1] - 6 * Math.sin(angle + 0.45)}" fill="none" stroke="${l.color}" stroke-width="1.5"/>`;
  }
  const placed = [];
  for (const r of rings) {
    const p = project(r.x, r.y);
    content += `<circle cx="${p[0]}" cy="${p[1]}" r="4" fill="white" stroke="${r.color}" stroke-width="1.8"/>`;
    const label = short(r.label);
    if (!label) continue;
    let y = p[1] - 10;
    if (
      placed.some((q) => Math.abs(q[0] - p[0]) < 70 && Math.abs(q[1] - y) < 18)
    )
      y = p[1] + 19;
    placed.push([p[0], y]);
    content += `<text x="${Math.max(45, Math.min(315, p[0]))}" y="${y}" fill="${r.color}" text-anchor="middle" font-size="11" font-family="Segoe UI,sans-serif" paint-order="stroke" stroke="white" stroke-width="4" stroke-linejoin="round">${label}</text>`;
  }
  svg.innerHTML = content;
  svg.setAttribute(
    "aria-label",
    `${title}. ${text} Geometry from iteration ${iteration}; this detail is independently scaled.`,
  );
}
