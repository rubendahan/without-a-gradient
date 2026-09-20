import { useEffect, useMemo, useRef, useState } from "react";
import { buildCity } from "../sim/network";
import { SignalPlan } from "../sim/plan";
import { evaluate, intersectionStats } from "../sim/delay";
import { MultiSwarm } from "../sim/pso";
import { fmtDelay } from "../lib/format";
import CityMap from "./CityMap";
import ConvergenceSparkline from "./ConvergenceSparkline";

type Mode = "equal" | "baseline" | "optimised";
const TARGET = 60;

export default function TrafficLab() {
  const [load, setLoad] = useState(0.55);
  return (
    <div className="traffic-experiment">
      <div className="demand-bar">
        <div>
          <label htmlFor="demand">Network demand</label>
          <span className="small-copy">
            Changing demand starts a new experiment.
          </span>
        </div>
        <div className="demand-input">
          <input
            id="demand"
            type="range"
            min="0.3"
            max="1.3"
            step="0.05"
            value={load}
            onChange={(e) => setLoad(+e.target.value)}
            aria-valuetext={`${load.toFixed(2)} demand multiplier`}
          />
          <div className="range-labels">
            <span>Light</span>
            <output htmlFor="demand">{load.toFixed(2)}×</output>
            <span>Heavy</span>
          </div>
        </div>
      </div>
      <Experiment key={load} load={load} />
    </div>
  );
}

function Experiment({ load }: { load: number }) {
  const { net, plan, baseline, baselineDelay } = useMemo(() => {
    const net = buildCity({ load, seed: 0 }),
      plan = new SignalPlan(net);
    const baseline = plan.proportional();
    return {
      net,
      plan,
      baseline,
      baselineDelay: evaluate(net, plan, baseline),
    };
  }, [load]);
  const [vector, setVector] = useState(baseline);
  const [mode, setMode] = useState<Mode>("baseline");
  const [running, setRunning] = useState(false);
  const [history, setHistory] = useState<number[]>([]);
  const [iter, setIter] = useState(0);
  const [animate, setAnimate] = useState(
    () => !window.matchMedia("(prefers-reduced-motion: reduce)").matches,
  );
  const swarm = useRef<MultiSwarm | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(
    () => () => {
      if (timer.current !== null) clearTimeout(timer.current);
    },
    [],
  );
  function stop() {
    if (timer.current !== null) clearTimeout(timer.current);
    setRunning(false);
  }
  function select(next: Mode) {
    stop();
    swarm.current = null;
    setIter(0);
    setHistory([]);
    setMode(next);
    setVector(next === "equal" ? plan.allGreen() : baseline);
  }
  function run() {
    if (running) {
      stop();
      return;
    }
    let sw = swarm.current;
    if (!sw || sw.iter >= TARGET) {
      sw = new MultiSwarm((x) => evaluate(net, plan, x), plan.dim, {
        seed: 0,
        init: baseline,
      });
      swarm.current = sw;
      setVector(sw.bestX.slice());
      setHistory(sw.history.slice());
      setIter(0);
    }
    const current = sw;
    setMode("optimised");
    setRunning(true);
    const tick = () => {
      if (document.hidden) {
        timer.current = setTimeout(tick, 150);
        return;
      }
      current.step();
      setVector(current.bestX.slice());
      setHistory(current.history.slice());
      setIter(current.iter);
      if (current.iter >= TARGET) setRunning(false);
      else timer.current = setTimeout(tick, 100);
    };
    timer.current = setTimeout(tick, 100);
  }
  const score = evaluate(net, plan, vector);
  const gain = (100 * (baselineDelay - score)) / baselineDelay;
  const stats = intersectionStats(net, plan, vector);
  const saturated = stats.filter((s) => s.sat >= 1).length;
  const maxSat = Math.max(...stats.map((s) => s.sat));
  const meanSat = stats.reduce((sum, s) => sum + s.sat, 0) / stats.length;
  const runText = running
    ? "Pause search"
    : iter > 0 && iter < TARGET
      ? "Resume search"
      : iter === TARGET
        ? "Run again"
        : "Optimise baseline";
  return (
    <div className="lab-grid">
      <div className="map-column">
        <div className="map-toolbar">
          <span>Synthetic grid / {net.intersections.length} junctions</span>
          <button onClick={() => setAnimate(!animate)} aria-pressed={animate}>
            {animate ? "Pause traffic" : "Animate traffic"}
          </button>
        </div>
        <div className="city-frame">
          <CityMap net={net} plan={plan} vector={vector} animate={animate} />
        </div>
        <div className="map-legend">
          <span>
            <i style={{ background: "#326da7" }} />
            Moving
          </span>
          <span>
            <i style={{ background: "#c65f43" }} />
            Waiting
          </span>
          <span>Signals: green / red</span>
          <span>Illustration · 16× time</span>
        </div>
      </div>
      <div className="lab-controls">
        <div>
          <p className="control-label">Compare plans</p>
          <div className="plan-options">
            <button
              aria-pressed={mode === "equal"}
              onClick={() => select("equal")}
            >
              Equal split
            </button>
            <button
              aria-pressed={mode === "baseline"}
              onClick={() => select("baseline")}
            >
              Demand-based
            </button>
          </div>
          <p className="plan-description">
            {mode === "equal"
              ? "Half the cycle per phase, with zero offsets."
              : mode === "baseline"
                ? "Green allocation follows critical flow ratios, with zero offsets."
                : "The best plan found, starting from the demand-based baseline."}
          </p>
        </div>
        <div className="metrics">
          <div>
            <span>Estimated total delay</span>
            <strong>{fmtDelay(score)}</strong>
            <small>vehicle-seconds / 4 hours</small>
          </div>
          <div>
            <span>Reduction vs baseline</span>
            <strong
              className={gain > 0.005 ? "good" : gain < -0.005 ? "bad" : ""}
            >
              {Math.abs(gain) < 0.005 ? "0.00" : gain.toFixed(2)}%
            </strong>
            <small>
              {gain < -0.005
                ? "Negative means more delay"
                : "Positive means less delay"}
            </small>
          </div>
        </div>
        <div className="search-actions">
          <button className="action-primary" onClick={run}>
            {runText} <span>{running ? "Ⅱ" : "→"}</span>
          </button>
          <button
            className="action-reset"
            onClick={() => select("baseline")}
            aria-label="Reset experiment to demand-based plan"
          >
            Reset
          </button>
        </div>
        <div className="run-progress">
          <progress
            value={iter}
            max={TARGET}
            aria-label="Search generations completed"
          />
          <div>
            <span>
              {iter} / {TARGET} generations
            </span>
            <span>
              {mode === "optimised" ? 48 * (iter + 1) : 0} search evaluations
            </span>
          </div>
        </div>
        <div className="convergence-box">
          <p className="control-label">Best delay relative to baseline</p>
          <ConvergenceSparkline history={history} sane={baselineDelay} />
          <p className="small-copy" role="status">
            {iter === TARGET
              ? "Run complete. The best plan stays on screen."
              : running
                ? "Searching. The best recorded delay can only decrease."
                : history.length > 1
                  ? "Search paused. Resume to keep the same run."
                  : "Run the search to draw the improvement curve."}
          </p>
        </div>
        <details className="network-details">
          <summary>{saturated} / 36 junctions at or above capacity</summary>
          <p>
            For the current plan, the busiest movement at each junction has mean
            saturation {meanSat.toFixed(2)} and maximum {maxSat.toFixed(2)}. A
            ratio of 1 means arrivals match green-time capacity. A network
            average can hide overloaded movements.
          </p>
        </details>
      </div>
    </div>
  );
}
