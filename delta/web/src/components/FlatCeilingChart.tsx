import { useEffect, useRef, useState } from "react";
import type { CeilingRow } from "../sim/analysis";

export default function FlatCeilingChart() {
  const [rows, setRows] = useState<CeilingRow[]>([]);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");
  const worker = useRef<Worker | null>(null);
  useEffect(() => () => worker.current?.terminate(), []);
  function calculate() {
    worker.current?.terminate();
    setRows([]);
    setRunning(true);
    setError("");
    const w = new Worker(new URL("../sim/sweep.worker.ts", import.meta.url), {
      type: "module",
    });
    worker.current = w;
    w.onmessage = (event) => {
      if (event.data.row) setRows((previous) => [...previous, event.data.row]);
      if (event.data.done || event.data.error) {
        setRunning(false);
        w.terminate();
      }
      if (event.data.error) setError(event.data.error);
    };
    w.onerror = () => {
      setRunning(false);
      setError("The calculation could not finish. Please try again.");
      w.terminate();
    };
    w.postMessage({});
  }
  const W = 840,
    H = 340,
    L = 60,
    R = 30,
    T = 38,
    B = 65;
  const ymax = Math.max(
    5,
    Math.ceil(Math.max(0, ...rows.map((r) => r.gainPct)) / 5) * 5,
  );
  const y = (v: number) => T + (1 - v / ymax) * (H - T - B);
  const band = (W - L - R) / 7;
  const ticks = Array.from({ length: 5 }, (_, i) => (ymax * i) / 4);
  return (
    <div className="sweep-panel">
      <div className="sweep-toolbar">
        <div>
          <strong>Delay reduction relative to the demand-based plan</strong>
          <p>Equal budget: 2,928 evaluations per demand level · seed 0</p>
        </div>
        <button
          className="action-primary"
          onClick={calculate}
          disabled={running}
        >
          {running
            ? `Computing ${rows.length} / 7`
            : rows.length
              ? "Run sweep again ↻"
              : "Calculate seven scenarios →"}
        </button>
      </div>
      {rows.length === 0 && !running ? (
        <div className="chart-empty">
          <span>Compare seven traffic loads</span>
          <p>
            Compare the reduction in estimated delay at seven demand levels.
            <br />
            Results are computed locally with a fixed evaluation budget.
          </p>
        </div>
      ) : (
        <div
          className="sweep-scroll"
          tabIndex={0}
          role="region"
          aria-label="Demand sweep results chart"
        >
          <svg
            viewBox={`0 0 ${W} ${H}`}
            role="img"
            aria-label="Percentage reduction in estimated delay at seven network demand levels. Exact values appear in the results table below."
          >
            {ticks.map((t) => (
              <g key={t}>
                <line
                  x1={L}
                  x2={W - R}
                  y1={y(t)}
                  y2={y(t)}
                  stroke="var(--color-line)"
                />
                <text
                  x={L - 12}
                  y={y(t) + 4}
                  textAnchor="end"
                  fontSize="12"
                  fill="var(--color-muted)"
                >
                  {t.toFixed(1)}%
                </text>
              </g>
            ))}
            {rows.map((r, i) => {
              const x = L + band * (i + 0.5);
              return (
                <g key={r.load}>
                  <title>
                    {r.load.toFixed(2)} demand: {r.gainPct.toFixed(2)}% less
                    delay
                  </title>
                  <rect
                    x={x - 24}
                    y={y(r.gainPct)}
                    width="48"
                    height={Math.max(1, y(0) - y(r.gainPct))}
                    fill="var(--color-accent)"
                  />
                  <text
                    x={x}
                    y={y(r.gainPct) - 10}
                    textAnchor="middle"
                    fontSize="13"
                    fill="var(--color-ink)"
                  >
                    {r.gainPct.toFixed(2)}%
                  </text>
                  <text
                    x={x}
                    y={H - B + 25}
                    textAnchor="middle"
                    fontSize="12"
                    fill="var(--color-muted)"
                  >
                    {r.load.toFixed(2)}×
                  </text>
                </g>
              );
            })}
            <text
              x={(W + L - R) / 2}
              y={H - 12}
              textAnchor="middle"
              fontSize="12"
              fill="var(--color-muted)"
            >
              Network demand multiplier
            </text>
          </svg>
        </div>
      )}
      <p className="small-copy" role="status">
        {error ||
          (running
            ? `Computed ${rows.length} of 7 scenarios. You can keep using the page.`
            : rows.length
              ? "A taller bar means a larger percentage improvement, not less total traffic delay. Results use one fixed seed per scenario."
              : "Calculated locally when you run the sweep.")}
      </p>
      {rows.length > 0 && (
        <details className="network-details">
          <summary>Exact values and comparison limits</summary>
          <div className="table-scroll">
            <table>
              <caption>
                Estimated total delay in vehicle-seconds over four hours.
              </caption>
              <thead>
                <tr>
                  <th>Demand</th>
                  <th>Baseline</th>
                  <th>Best found</th>
                  <th>Reduction</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.load}>
                    <td>{r.load.toFixed(2)}×</td>
                    <td>{Math.round(r.sane).toLocaleString("en")}</td>
                    <td>{Math.round(r.optimized).toLocaleString("en")}</td>
                    <td>{r.gainPct.toFixed(2)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p>
            Each scenario uses the same generated network structure and random
            seed. Demand changes the objective, so compare percentage gains
            against each scenario’s own baseline. Multiple seeds and matched
            evaluation budgets are needed for a broader performance study.
          </p>
        </details>
      )}
    </div>
  );
}
