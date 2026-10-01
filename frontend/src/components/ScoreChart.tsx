import type { IterationRecord } from "../types";

interface Props {
  history: IterationRecord[];
  maxRounds: number;
}

// viewBox units; the SVG scales to its container width.
const W = 640;
const H = 200;
const PAD = { top: 12, right: 28, bottom: 28, left: 40 };
// Gap between the y-axis labels and round 1, so its spread-out dots don't touch them.
const INSET = 28;

/** Each candidate's train score as a dot, plus a best-so-far line with a soft fill.
 *  Hand-rolled SVG: it's one small chart, not worth a charting dependency. */
export function ScoreChart({ history, maxRounds }: Props) {
  const rounds = Math.max(maxRounds, history.length, 2);
  const x = (round: number) =>
    PAD.left + INSET + (round / (rounds - 1)) * (W - PAD.left - INSET - PAD.right);
  const y = (score: number) => PAD.top + (1 - score) * (H - PAD.top - PAD.bottom);
  const line = history.map((r) => `${x(r.iteration)},${y(r.best_score)}`).join(" ");
  const area =
    history.length > 1
      ? `M${x(0)},${y(0)} L${line.replaceAll(" ", " L")} L${x(history.length - 1)},${y(0)} Z`
      : "";

  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Score by round">
      <defs>
        <linearGradient id="area-fill" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor="var(--accent)" stopOpacity="0.25" />
          <stop offset="100%" stopColor="var(--accent)" stopOpacity="0" />
        </linearGradient>
      </defs>
      {[0, 0.5, 1].map((t) => (
        <g key={t}>
          <line className="grid" x1={PAD.left} x2={W - PAD.right} y1={y(t)} y2={y(t)} />
          <text className="axis" x={PAD.left - 8} y={y(t) + 4} textAnchor="end">
            {Math.round(t * 100)}%
          </text>
        </g>
      ))}
      {Array.from({ length: rounds }, (_, i) => (
        <text key={i} className="axis" x={x(i)} y={H - 8} textAnchor="middle">
          R{i + 1}
        </text>
      ))}
      {area && <path d={area} fill="url(#area-fill)" />}
      {history.flatMap((r) =>
        r.evals.map((e, j) => (
          <circle
            key={`${r.iteration}-${e.candidate_id}`}
            className={e.candidate_id === "baseline" ? "dot baseline" : "dot"}
            // Spread a round's dots slightly so equal scores don't hide each other.
            cx={x(r.iteration) + (j - (r.evals.length - 1) / 2) * 7}
            cy={y(e.mean_score)}
            r={4}
          >
            <title>{`${e.candidate_id}: ${Math.round(e.mean_score * 100)}%`}</title>
          </circle>
        )),
      )}
      {history.length > 0 && <polyline className="best-line" points={line} />}
      {history.map((r) => (
        <circle key={r.iteration} className="best-dot" cx={x(r.iteration)} cy={y(r.best_score)} r={5} />
      ))}
    </svg>
  );
}
