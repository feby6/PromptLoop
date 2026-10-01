import type { Candidate, IterationRecord, LLMUsage, RunEvent } from "../types";
import { ArrowLeftIcon, XIcon } from "./Icons";
import { ScoreChart } from "./ScoreChart";

interface Props {
  events: RunEvent[];
  maxRounds: number;
  error: string | null;
  onCancel: () => void;
  onBack: () => void;
}

const pct = (score: number) => `${Math.round(score * 100)}%`;

/** Live view of a run, derived entirely from the event stream. */
export function RunProgress({ events, maxRounds, error, onCancel, onBack }: Props) {
  const history = events
    .filter((e) => e.type === "iteration")
    .map((e) => e.data.record as IterationRecord);
  const usage = [...events].reverse().find((e) => e.data.usage)?.data.usage as
    | LLMUsage
    | undefined;
  const latestCandidates = [...events].reverse().find((e) => e.type === "candidates")?.data
    .candidates as Candidate[] | undefined;
  const latest = events.at(-1);
  const stopped = error !== null || (latest !== undefined && ["error", "cancelled"].includes(latest.type));
  const best = history.at(-1)?.best_score;
  const round = Math.min(history.length + (stopped ? 0 : 1), maxRounds);

  // Scores for the candidates currently on screen, once their round has been scored.
  const scores = new Map<string, number>();
  for (const r of history) for (const e of r.evals) scores.set(e.candidate_id, e.mean_score);

  return (
    <div className="panel">
      <section className="section">
        <div className="live-head">
          <div>
            <h2>{stopped ? "Run stopped" : "Optimising your prompt"}</h2>
            <p className={`status${error ? " error-text" : ""}`}>
              {!stopped && <span className="spinner" />}
              {error ?? latest?.message ?? "Starting…"}
            </p>
          </div>
          <div className="live-score">
            <span className="muted small">Best training score</span>
            <strong>{best === undefined ? "–" : pct(best)}</strong>
          </div>
        </div>
        <div className="progress" aria-label={`Round ${round} of up to ${maxRounds}`}>
          <div className="progress-bar" style={{ width: `${(history.length / maxRounds) * 100}%` }} />
        </div>
        <p className="muted small">
          Round {round} of up to {maxRounds}
          {usage && ` · ${usage.calls} calls to your model so far`}
        </p>
        <ScoreChart history={history} maxRounds={maxRounds} />
      </section>

      {latestCandidates && latestCandidates.length > 0 && (
        <section className="section">
          <h3>Prompts being tried</h3>
          <div className="candidates">
            {latestCandidates.map((c) => {
              const score = scores.get(c.id);
              return (
                <article key={c.id} className="candidate">
                  <header>
                    <span className="tag">{c.id === "baseline" ? "your task, as-is" : c.id}</span>
                    <span className={`score-pill${score === undefined ? " pending" : ""}`}>
                      {score === undefined ? "scoring…" : pct(score)}
                    </span>
                  </header>
                  <p>{c.prompt}</p>
                </article>
              );
            })}
          </div>
        </section>
      )}

      <details className="section log-details">
        <summary>Activity log ({events.length})</summary>
        <ol className="log">
          {events.map((e) => (
            <li key={e.seq} className={`log-${e.type}`}>
              {e.message}
            </li>
          ))}
        </ol>
      </details>

      <div className="panel-actions">
        {stopped ? (
          <button type="button" className="btn ghost" onClick={onBack}>
            <ArrowLeftIcon /> Back to examples
          </button>
        ) : (
          <button type="button" className="btn danger-ghost" onClick={onCancel}>
            <XIcon /> Cancel run
          </button>
        )}
      </div>
    </div>
  );
}
