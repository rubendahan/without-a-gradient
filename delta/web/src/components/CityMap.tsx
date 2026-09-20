import { useEffect, useRef } from "react";
import type { RoadNetwork } from "../sim/network";
import { totalArrival } from "../sim/network";
import { rngFrom } from "../sim/rng";
import type { SignalPlan } from "../sim/plan";

interface Props {
  net: RoadNetwork;
  plan: SignalPlan;
  vector: number[];
  animate: boolean;
}

// The city as a small, legible traffic simulation. Streets carry cars that drive
// at a steady speed and stop at red lights, so queues build up wherever a signal
// cannot clear its demand. The lights cycle in real time from the current plan:
// each junction is green for the main street for a fraction of its cycle (the
// green split) and shifted by its offset, so well-coordinated offsets send a
// platoon through several greens in a row (a green wave). When you optimise, you
// watch the queues shrink and the waves line up.

interface Car {
  s: number; // position along the lane, 0..1
  prevS: number;
}
interface Lane {
  nodes: { k: number; s: number }[]; // intersections on this lane, sorted along it
  demand: number; // cars per second to spawn, from the lane's demand
  v: number; // speed in lane-fraction per second
  line: number; // row index (h lanes) or col index (v lanes)
  cars: Car[];
}

const SPEEDUP = 16; // compress real seconds so a 90 s cycle plays in a few seconds
const STOP = 0.02; // gap a car keeps before a red light, in lane fraction
const GAP = 0.024; // gap a car keeps behind the car ahead
const SPAWN = 7; // spawn-rate scale
const MAX_CARS = 42;

const frac = (x: number) => x - Math.floor(x);

export default function CityMap({ net, plan, vector, animate }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const props = useRef({ net, plan, vector, animate });
  useEffect(() => {
    props.current = { net, plan, vector, animate };
  }, [net, plan, vector, animate]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    let raf = 0;
    let last = performance.now();
    let simTime = 0;
    let rand = rngFrom(0);
    let visible = true;
    const io = new IntersectionObserver((entries) => {
      visible = entries[0].isIntersecting;
      last = performance.now();
    });
    io.observe(canvas);
    let builtFor: RoadNetwork | null = null;
    let rows: Lane[] = [];
    let cols: Lane[] = [];

    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.floor(rect.width * dpr);
      canvas.height = Math.floor(rect.height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);

    const buildLanes = (net: RoadNetwork) => {
      const maxCol = Math.max(1, net.width - 1);
      const maxRow = Math.max(1, net.height - 1);
      const indexed = net.intersections.map((it, k) => ({ it, k }));
      const ff = net.freeFlowTravelTime;
      rows = [];
      for (let r = 0; r < net.height; r++) {
        const here = indexed
          .filter((e) => e.it.row === r)
          .sort((a, b) => a.it.col - b.it.col);
        if (here.length < 2) continue;
        const demand =
          here.reduce((sum, e) => sum + totalArrival(e.it.phases[0]), 0) /
          here.length;
        rows.push({
          nodes: here.map((e) => ({ k: e.k, s: e.it.col / maxCol })),
          demand,
          v: SPEEDUP / (maxCol * ff),
          line: r,
          cars: [],
        });
      }
      cols = [];
      for (let c = 0; c < net.width; c++) {
        const here = indexed
          .filter((e) => e.it.col === c)
          .sort((a, b) => a.it.row - b.it.row);
        if (here.length < 2) continue;
        const demand =
          here.reduce((sum, e) => sum + totalArrival(e.it.phases[1]), 0) /
          here.length;
        cols.push({
          nodes: here.map((e) => ({ k: e.k, s: e.it.row / maxRow })),
          demand,
          v: SPEEDUP / (maxRow * ff),
          line: c,
          cars: [],
        });
      }
      rand = rngFrom(0);
      simTime = 0;
      builtFor = net;
    };

    const advance = (
      lane: Lane,
      dt: number,
      isGreen: (k: number) => boolean,
    ) => {
      lane.cars.sort((a, b) => a.s - b.s);
      for (let i = lane.cars.length - 1; i >= 0; i--) {
        const car = lane.cars[i];
        let limit = 1.06;
        for (const nd of lane.nodes) {
          if (nd.s > car.s + 1e-4 && !isGreen(nd.k)) {
            limit = Math.min(limit, nd.s - STOP);
            break;
          }
        }
        if (i < lane.cars.length - 1)
          limit = Math.min(limit, lane.cars[i + 1].s - GAP);
        let ns = car.s + lane.v * dt;
        if (ns > limit) ns = limit;
        if (ns < car.s) ns = car.s;
        car.prevS = car.s;
        car.s = ns;
      }
      lane.cars = lane.cars.filter((c) => c.s <= 1.05);
      const minS = lane.cars.length
        ? Math.min(...lane.cars.map((c) => c.s))
        : 1;
      if (
        minS > GAP * 1.5 &&
        lane.cars.length < MAX_CARS &&
        rand() < lane.demand * SPAWN * dt
      ) {
        lane.cars.push({ s: 0, prevS: 0 });
      }
    };

    const draw = (now: number) => {
      const { net, plan, vector, animate } = props.current;
      if (!visible || document.hidden) {
        last = now;
        raf = requestAnimationFrame(draw);
        return;
      }
      if (net !== builtFor) buildLanes(net);
      let dt = (now - last) / 1000;
      last = now;
      if (dt > 0.05) dt = 0.05; // keep the sim stable if the tab was backgrounded

      const rect = canvas.getBoundingClientRect();
      const W = rect.width;
      const H = rect.height;
      const pad = 30;
      const maxCol = Math.max(1, net.width - 1);
      const maxRow = Math.max(1, net.height - 1);
      const xFromS = (s: number) => pad + s * (W - 2 * pad);
      const yFromS = (s: number) => pad + s * (H - 2 * pad);
      const xLane = (c: number) => pad + (c / maxCol) * (W - 2 * pad);
      const yLane = (r: number) => pad + (r / maxRow) * (H - 2 * pad);

      // signal state now: green[k] true means the main (horizontal) street is green
      const tm = plan.fromVector(vector);
      if (animate) simTime += dt * SPEEDUP;
      const T = simTime;
      const green = net.intersections.map(
        (_it, k) =>
          frac((T - tm[k].offset) / tm[k].cycleLength) < tm[k].splits[0],
      );

      // step the simulation
      if (animate) {
        for (const lane of rows) advance(lane, dt, (k) => green[k]);
        for (const lane of cols) advance(lane, dt, (k) => !green[k]);
      }

      ctx.clearRect(0, 0, W, H);

      ctx.fillStyle = "#f0f4f7";
      for (let r = 0; r < maxRow; r++)
        for (let c = 0; c < maxCol; c++) {
          roundRect(
            ctx,
            xLane(c) + 12,
            yLane(r) + 12,
            Math.max(1, (W - 2 * pad) / maxCol - 24),
            Math.max(1, (H - 2 * pad) / maxRow - 24),
            5,
          );
          ctx.fill();
        }
      ctx.fillStyle = "#60758a";
      ctx.font = "10px Segoe UI, sans-serif";
      ctx.textAlign = "center";
      for (let c = 0; c <= maxCol; c++)
        ctx.fillText(String(c + 1), xLane(c), 14);
      for (let r = 0; r <= maxRow; r++)
        ctx.fillText(String.fromCharCode(65 + r), 12, yLane(r) + 3);
      ctx.textAlign = "start";
      // streets
      ctx.lineCap = "round";
      ctx.strokeStyle = "#dce4eb";
      ctx.lineWidth = 9;
      for (const lane of rows) {
        const y = yLane(lane.line);
        ctx.beginPath();
        ctx.moveTo(xFromS(0), y);
        ctx.lineTo(xFromS(1), y);
        ctx.stroke();
      }
      for (const lane of cols) {
        const x = xLane(lane.line);
        ctx.beginPath();
        ctx.moveTo(x, yFromS(0));
        ctx.lineTo(x, yFromS(1));
        ctx.stroke();
      }

      // cars: warm when moving, red when stopped (a stopped cluster reads as a jam)
      const drawCar = (
        x: number,
        y: number,
        horizontal: boolean,
        stopped: boolean,
      ) => {
        ctx.save();
        ctx.fillStyle = stopped ? "#c65f43" : "#326da7";
        ctx.shadowColor = stopped ? "#ff5a5a" : "#ffb02e";
        ctx.shadowBlur = 0;
        const w = horizontal ? 8 : 4.4;
        const h = horizontal ? 4.4 : 8;
        roundRect(ctx, x - w / 2, y - h / 2, w, h, 1.4);
        ctx.fill();
        ctx.restore();
      };
      const moved = (c: Car, v: number) => c.s - c.prevS > v * dt * 0.35;
      for (const lane of rows) {
        const y = yLane(lane.line) + 3.4; // drive on the right of the centreline
        for (const c of lane.cars)
          drawCar(xFromS(c.s), y, true, !moved(c, lane.v));
      }
      for (const lane of cols) {
        const x = xLane(lane.line) - 3.4;
        for (const c of lane.cars)
          drawCar(x, yFromS(c.s), false, !moved(c, lane.v));
      }

      // junctions with their signal faces
      net.intersections.forEach((it, k) => {
        const x = xLane(it.col);
        const y = yLane(it.row);
        const mainGreen = green[k];
        ctx.fillStyle = "#ffffff";
        ctx.strokeStyle = "#b5c4d2";
        ctx.lineWidth = 1;
        roundRect(ctx, x - 7, y - 7, 14, 14, 3);
        ctx.fill();
        ctx.stroke();
        const G = "#278362";
        const R = "#c65f43";
        // horizontal faces (left, right) show the main-street light
        const hC = mainGreen ? G : R;
        const vC = mainGreen ? R : G;
        ctx.fillStyle = hC;
        ctx.fillRect(x - 7, y - 2.2, 2.2, 4.4);
        ctx.fillRect(x + 4.8, y - 2.2, 2.2, 4.4);
        ctx.fillStyle = vC;
        ctx.fillRect(x - 2.2, y - 7, 4.4, 2.2);
        ctx.fillRect(x - 2.2, y + 4.8, 4.4, 2.2);
      });

      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      io.disconnect();
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="h-full w-full"
      role="img"
      aria-label="Synthetic 36-junction grid. Cars illustrate red-light waiting; the reported delay is calculated by a separate analytical model."
    />
  );
}

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}
