interface Props {
  history: number[];
  sane: number;
}
export default function ConvergenceSparkline({ history, sane }: Props) {
  const W = 420,
    H = 150,
    left = 46,
    right = 18,
    top = 20,
    bottom = 30;
  const values = history.map((v) => (100 * v) / sane);
  const low = Math.floor(Math.min(99, ...values) - 0.3);
  const high = 100.4;
  const x = (i: number) => left + (i / 60) * (W - left - right);
  const y = (v: number) =>
    top + ((high - v) / (high - low)) * (H - top - bottom);
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-label={
        values.length
          ? `Best delay fell from ${values[0].toFixed(2)} to ${values.at(-1)!.toFixed(2)} percent of baseline over ${values.length - 1} generations.`
          : "Search progress. The horizontal dashed line marks 100 percent of baseline delay."
      }
    >
      {[low, (low + 100) / 2, 100].map((v) => (
        <g key={v}>
          <line
            x1={left}
            x2={W - right}
            y1={y(v)}
            y2={y(v)}
            stroke="var(--color-line)"
            strokeDasharray={v === 100 ? "4 4" : undefined}
          />
          <text
            x={left - 8}
            y={y(v) + 4}
            textAnchor="end"
            fontSize="10"
            fill="var(--color-muted)"
          >
            {v.toFixed(1)}%
          </text>
        </g>
      ))}
      <polyline
        points={values.map((v, i) => `${x(i)},${y(v)}`).join(" ")}
        fill="none"
        stroke="var(--color-accent)"
        strokeWidth="2.5"
        strokeLinejoin="round"
      />
      {values.length > 0 && (
        <circle
          cx={x(values.length - 1)}
          cy={y(values.at(-1)!)}
          r="3"
          fill="var(--color-accent)"
        />
      )}
      {[0, 20, 40, 60].map((v) => (
        <text
          key={v}
          x={x(v)}
          y={H - 12}
          textAnchor="middle"
          fontSize="10"
          fill="var(--color-muted)"
        >
          {v}
        </text>
      ))}
      <text
        x={W - right}
        y="12"
        textAnchor="end"
        fontSize="10"
        fill="var(--color-muted)"
      >
        100% = baseline · generation →
      </text>
    </svg>
  );
}
