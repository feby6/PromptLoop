import type { AppConfig, EditableExample, RunSettings, ScorerChoice } from "../types";
import { ExamplesEditor, isComplete } from "./ExamplesEditor";
import { ArrowLeftIcon, ArrowRightIcon, SparkleIcon } from "./Icons";

interface Props {
  config: AppConfig;
  examples: EditableExample[];
  onExamplesChange: (examples: EditableExample[]) => void;
  settings: RunSettings;
  onSettingsChange: (settings: RunSettings) => void;
  busy: boolean;
  onGenerateMore: () => void;
  onBack: () => void;
  onStart: () => void;
}

const SCORERS: { value: ScorerChoice; label: string; hint: string }[] = [
  { value: "auto", label: "Auto", hint: "Picked from your expected outputs" },
  { value: "exact", label: "Exact match", hint: "Labels and short answers" },
  { value: "json_match", label: "JSON fields", hint: "Structured extraction" },
  { value: "judge", label: "LLM judge", hint: "Free-form text" },
];

// Mirrors the backend's split (api/runs.py + config synthesis.val_fraction).
const VAL_FRACTION = 0.3;

/** Upper bound on calls to the user's model, so the cost is clear before starting. */
function estimateCalls(total: number, s: RunSettings): number {
  const nVal = Math.max(2, Math.round(total * VAL_FRACTION));
  const nTrain = Math.max(1, total - nVal);
  const firstRound = 1 + (s.nCandidates + 1) * nTrain;
  const laterRounds = (s.maxIterations - 1) * (2 + s.nCandidates * nTrain);
  return firstRound + laterRounds + 2 * nVal;
}

export function ReviewStep(props: Props) {
  const { config, examples, settings, busy } = props;
  const complete = examples.filter(isComplete).length;
  const unchecked = examples.filter((e) => e.generated).length;
  const enough = complete >= config.min_examples;
  const set = (patch: Partial<RunSettings>) => props.onSettingsChange({ ...settings, ...patch });

  return (
    <div className="panel">
      <section className="section">
        <div className="section-head">
          <div>
            <h2>Check your examples</h2>
            <p className="muted">
              The score is only as good as these answers. Fix or remove anything wrong.
            </p>
          </div>
          <button
            type="button"
            className="btn ghost"
            onClick={props.onGenerateMore}
            disabled={busy || examples.length >= config.max_examples}
          >
            {busy ? <span className="spinner" /> : <SparkleIcon />}
            {busy ? "Drafting…" : "Draft 5 more"}
          </button>
        </div>
        <div className="summary">
          <div>
            <strong>{complete}</strong> examples
          </div>
          <div className={unchecked ? "warn" : ""}>
            <strong>{unchecked}</strong> drafted, not yet edited
          </div>
          <div>
            <strong>~{Math.max(2, Math.round(complete * VAL_FRACTION))}</strong> held out for the
            final score
          </div>
        </div>
        <ExamplesEditor
          examples={examples}
          onChange={props.onExamplesChange}
          maxExamples={config.max_examples}
        />
      </section>

      <section className="section">
        <h2>Run settings</h2>
        <p className="muted">The defaults suit most tasks.</p>
        <div className="field">
          <span>How to score answers</span>
          <div className="option-grid">
            {SCORERS.map((s) => (
              <button
                key={s.value}
                type="button"
                className={`option${settings.scorer === s.value ? " active" : ""}`}
                onClick={() => set({ scorer: s.value })}
              >
                <strong>{s.label}</strong>
                <span>{s.hint}</span>
              </button>
            ))}
          </div>
        </div>
        {settings.scorer === "judge" && (
          <label className="field">
            <span>Grading rubric (optional)</span>
            <textarea
              rows={2}
              value={settings.rubric}
              onChange={(e) => set({ rubric: e.target.value })}
              placeholder="What makes an answer correct? Leave empty for a sensible default."
            />
          </label>
        )}
        <div className="grid-2">
          <label className="field">
            <span>
              New prompts per round <em>{settings.nCandidates}</em>
            </span>
            <input
              type="range"
              min={1}
              max={5}
              value={settings.nCandidates}
              onChange={(e) => set({ nCandidates: Number(e.target.value) })}
            />
          </label>
          <label className="field">
            <span>
              Max rounds <em>{settings.maxIterations}</em>
            </span>
            <input
              type="range"
              min={1}
              max={8}
              value={settings.maxIterations}
              onChange={(e) => set({ maxIterations: Number(e.target.value) })}
            />
          </label>
        </div>
        <p className="note">
          Up to ~{estimateCalls(complete, settings)} calls to your model. Runs usually stop
          earlier, once the score stops improving.
        </p>
      </section>

      <div className="panel-actions">
        <button type="button" className="btn ghost" onClick={props.onBack} disabled={busy}>
          <ArrowLeftIcon /> Back
        </button>
        <button type="button" className="btn primary" onClick={props.onStart} disabled={!enough || busy}>
          Start optimising <ArrowRightIcon />
        </button>
      </div>
      {!enough && (
        <p className="error-text right">Needs at least {config.min_examples} complete examples.</p>
      )}
    </div>
  );
}
