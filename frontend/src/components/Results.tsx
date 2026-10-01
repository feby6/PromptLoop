import { useState } from "react";
import type { DiffChunk, LineageStep, RunResult } from "../types";
import { CheckIcon, CopyIcon, DownloadIcon } from "./Icons";

const STOP_TEXT: Record<RunResult["stop_reason"], string> = {
  target_reached: "reached a perfect training score",
  max_iterations: "used every round",
  plateau: "stopped improving",
};

const pct = (score: number | null) => (score === null ? "–" : `${Math.round(score * 100)}%`);

function DiffView({ chunks }: { chunks: DiffChunk[] }) {
  return (
    <p className="diff">
      {chunks.map((c, i) =>
        c.op === "insert" ? (
          <ins key={i}>{c.text}</ins>
        ) : c.op === "delete" ? (
          <del key={i}>{c.text}</del>
        ) : (
          <span key={i}>{c.text}</span>
        ),
      )}
    </p>
  );
}

function ScoreBar({ label, score, tone }: { label: string; score: number | null; tone: "muted" | "accent" }) {
  return (
    <div className="score-bar">
      <div className="score-bar-head">
        <span>{label}</span>
        <strong>{pct(score)}</strong>
      </div>
      <div className="track">
        <div className={`fill ${tone}`} style={{ width: `${(score ?? 0) * 100}%` }} />
      </div>
    </div>
  );
}

type Tab = "evolution" | "failures" | "leaderboard";

interface Props {
  result: RunResult;
  lineage: LineageStep[];
  onEditAndRerun: () => void;
  onStartOver: () => void;
}

export function Results({ result, lineage, onEditAndRerun, onStartOver }: Props) {
  const [copied, setCopied] = useState(false);
  const [tab, setTab] = useState<Tab>(lineage.length > 1 ? "evolution" : "failures");
  const failures = result.val_eval.results.filter((r) => r.score < 1);
  const delta =
    result.baseline_val_score === null ? null : result.val_score - result.baseline_val_score;
  const deltaPts = delta === null ? null : Math.round(delta * 100);

  // Every scored prompt across all rounds, best first.
  const prompts = new Map(result.history.flatMap((r) => r.candidates.map((c) => [c.id, c])));
  const leaderboard = result.history
    .flatMap((r) => r.evals.map((e) => ({ id: e.candidate_id, round: r.iteration + 1, score: e.mean_score })))
    .sort((a, b) => b.score - a.score);

  const copy = async () => {
    await navigator.clipboard.writeText(result.best_candidate.prompt);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };
  const download = () => {
    const blob = new Blob([result.best_candidate.prompt], { type: "text/plain" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "prompt.txt";
    a.click();
    URL.revokeObjectURL(a.href);
  };

  return (
    <div className="panel">
      <section className="section hero-result">
        <div className="hero-copy">
          <span className="eyebrow">Result on examples the optimiser never saw</span>
          <h2>
            {deltaPts === null
              ? `Your optimised prompt scores ${pct(result.val_score)}`
              : deltaPts > 0
                ? `${deltaPts} points better than your task description alone`
                : deltaPts === 0
                  ? "As good as your task description alone"
                  : `${-deltaPts} points worse than your task description alone`}
          </h2>
          <p className="muted">
            {result.n_val} held-out and {result.n_train} training examples · {result.history.length}{" "}
            round{result.history.length === 1 ? "" : "s"}, {STOP_TEXT[result.stop_reason]} ·{" "}
            {result.usage.calls} calls to <code>{result.target_model}</code>
            {result.n_val < 5 && ". With this few held-out examples, treat the score as rough."}
          </p>
        </div>
        <div className="hero-bars">
          <ScoreBar label="Your task description, as-is" score={result.baseline_val_score} tone="muted" />
          <ScoreBar label="Optimised prompt" score={result.val_score} tone="accent" />
        </div>
      </section>

      <section className="section">
        <div className="section-head">
          <h2>Your optimised prompt</h2>
          <div className="row-gap">
            <button type="button" className="btn ghost small" onClick={download}>
              <DownloadIcon /> .txt
            </button>
            <button type="button" className="btn primary small" onClick={copy}>
              {copied ? <CheckIcon /> : <CopyIcon />} {copied ? "Copied" : "Copy"}
            </button>
          </div>
        </div>
        <pre className="prompt">{result.best_candidate.prompt}</pre>
      </section>

      <section className="section">
        <div className="tabs" role="tablist">
          {(
            [
              ["evolution", `How it evolved (${Math.max(lineage.length - 1, 0)})`],
              ["failures", `Still failing (${failures.length})`],
              ["leaderboard", `All prompts (${leaderboard.length})`],
            ] as [Tab, string][]
          ).map(([id, label]) => (
            <button
              key={id}
              type="button"
              role="tab"
              aria-selected={tab === id}
              className={tab === id ? "active" : ""}
              onClick={() => setTab(id)}
            >
              {label}
            </button>
          ))}
        </div>

        {tab === "evolution" &&
          (lineage.length > 1 ? (
            <ol className="timeline">
              {lineage.slice(1).map((step, i) => (
                <li key={step.candidate.id}>
                  <div className="timeline-head">
                    <span className="tag">{lineage[i].candidate.id}</span> →{" "}
                    <span className="tag accent">{step.candidate.id}</span>
                    <span className="muted small">training {pct(step.train_score)}</span>
                  </div>
                  <DiffView chunks={step.diff_from_parent} />
                </li>
              ))}
            </ol>
          ) : (
            <p className="empty">The winner came from the first round, so there is no edit history.</p>
          ))}

        {tab === "failures" &&
          (failures.length === 0 ? (
            <p className="empty">Every held-out example passed.</p>
          ) : (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Input</th>
                    <th>Expected</th>
                    <th>Got</th>
                    <th className="num">Score</th>
                  </tr>
                </thead>
                <tbody>
                  {failures.map((f, i) => (
                    <tr key={i}>
                      <td>{f.input}</td>
                      <td>
                        <code>{f.expected_output}</code>
                      </td>
                      <td>
                        {f.error ? <em className="error-text">{f.error}</em> : <code>{f.output}</code>}
                      </td>
                      <td className="num">{pct(f.score)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ))}

        {tab === "leaderboard" && (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Prompt</th>
                  <th className="num">Round</th>
                  <th className="num">Training</th>
                </tr>
              </thead>
              <tbody>
                {leaderboard.map((row) => (
                  <tr key={row.id} className={row.id === result.best_candidate.id ? "winner" : ""}>
                    <td>
                      <span className="tag">{row.id === "baseline" ? "task as-is" : row.id}</span>
                      <span className="clamp">{prompts.get(row.id)?.prompt}</span>
                    </td>
                    <td className="num">{row.round}</td>
                    <td className="num">{pct(row.score)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <div className="panel-actions">
        <button type="button" className="btn ghost" onClick={onEditAndRerun}>
          Edit examples and rerun
        </button>
        <button type="button" className="btn primary" onClick={onStartOver}>
          Start a new task
        </button>
      </div>
    </div>
  );
}
